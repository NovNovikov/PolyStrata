# tools/dspark_lucebox_remap.py - переписать DSpark drafter GGUF, собранный конвертером Lucebox
# (server/scripts/convert_ds4_dspark_draft_to_gguf.py, архитектура deepseek41-dflash-draft), в
# unsloth-раскладку dflash, которую читает upstream tools/dsv41/pack_gguf.py без правок.
#
# Что меняется (только заголовок; байты весов идут как есть, id типов ggml у Lucebox совпадают
# с таблицей TYPES пакера: F32 0, F16 1, Q2_K 10, Q4_K 12, MXFP4 39):
#   general.architecture          deepseek41-dflash-draft -> dflash
#   kv                            dflash.block_count / block_size / expert_count /
#                                 expert_used_count / target_layers плоские, как в unsloth-файле
#                                 (у Lucebox геометрия под <arch>.* и вложенно <arch>.dflash.*)
#   dflash.fc.weight              -> fc.weight
#   dflash.hidden_norm.weight     -> enc.output_norm.weight
#   output_norm.weight            -> как есть
#   dflash.dspark.markov.w1 / w2  -> markov_w1.weight / markov_w2.weight
#   dflash.dspark.confidence.weight -> conf_proj.weight
#   dflash.dspark.confidence.bias -> выбрасывается: V4.1 не несёт bias у confidence_head
#                                 (конвертер Lucebox пишет нулевой), unsloth-пак его не содержит
#
# Проверка перед доверием: имена blk.N.* у Lucebox и unsloth общие (llama.cpp DFLASH), стеки
# экспертов [in][out][n_exp] одинакового порядка, блок 5, mask 128799, rank 256, цели 37..39.
#
# запуск:  python tools/dspark_lucebox_remap.py <lucebox-dspark.gguf> <out.gguf>
import pathlib
import struct
import sys

ARCH_SRC = "deepseek41-dflash-draft"
TYPES = {0: (1, 4), 1: (1, 2), 8: (32, 34), 10: (256, 84), 11: (256, 110), 12: (256, 144),
         13: (256, 176), 14: (256, 210), 19: (256, 50), 20: (32, 18), 21: (256, 110),
         22: (256, 82), 23: (256, 136), 29: (256, 56), 30: (1, 2), 39: (32, 17),
         43: (32, 33), 67: (32, 33)}          # то же, что tools/dsv41/pack_gguf.py TYPES
HEAD = {"dflash.fc.weight": "fc.weight", "dflash.hidden_norm.weight": "enc.output_norm.weight",
        "output_norm.weight": "output_norm.weight", "dflash.dspark.markov.w1": "markov_w1.weight",
        "dflash.dspark.markov.w2": "markov_w2.weight",
        "dflash.dspark.confidence.weight": "conf_proj.weight"}
DROP = {"dflash.dspark.confidence.bias"}
FMT = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d",
       13: "e", 14: "H"}


def read_gguf(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("%s: not a GGUF file" % path)
        ver, nt, nkv = struct.unpack("<IQQ", f.read(20))
        kv = {}

        def rstr():
            n = struct.unpack("<Q", f.read(8))[0]
            return f.read(n).decode("utf-8", "replace")

        def rval(t):
            if t in FMT:
                n = struct.calcsize("<" + FMT[t])
                return struct.unpack("<" + FMT[t], f.read(n))[0]
            if t == 8:
                return rstr()
            if t == 9:
                at, n = struct.unpack("<IQ", f.read(12))
                return [rval(at) for _ in range(n)]
            raise SystemExit("%s: metadata type %d" % (path, t))
        for _ in range(nkv):
            k = rstr()
            kv[k] = rval(struct.unpack("<I", f.read(4))[0])
        tensors = []
        for _ in range(nt):
            name = rstr()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
            typ, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, typ, dims, off))
        align = int(kv.get("general.alignment", 32))
        data_start = (f.tell() + align - 1) // align * align
    return kv, tensors, data_start


def nbytes(dims, typ):
    per, bpw = TYPES[typ]
    n = 1
    for d in dims:
        n *= d
    return n // per * bpw


def w_str(f, s):
    b = s.encode("utf-8")
    f.write(struct.pack("<Q", len(b)))
    f.write(b)


def w_kv(f, key, value):
    w_str(f, key)
    if isinstance(value, bool):
        f.write(struct.pack("<IB", 7, int(value)))
    elif isinstance(value, str):
        f.write(struct.pack("<I", 8))
        w_str(f, value)
    elif isinstance(value, float):
        f.write(struct.pack("<If", 6, value))
    elif isinstance(value, list):
        et, fmt = (5, "i") if all(isinstance(v, int) for v in value) else (6, "f")
        f.write(struct.pack("<IIQ", 9, et, len(value)))
        f.write(struct.pack("<%d%s" % (len(value), fmt), *value))
    else:
        f.write(struct.pack("<Ii", 5, int(value)))


def main():
    src, dst = pathlib.Path(sys.argv[1]).resolve(), pathlib.Path(sys.argv[2]).resolve()
    kv, tensors, data_start = read_gguf(src)
    if kv.get("general.architecture") != ARCH_SRC:
        raise SystemExit("%s: general.architecture is %r, expected %r"
                         % (src, kv.get("general.architecture"), ARCH_SRC))
    g = lambda k: kv["%s.%s" % (ARCH_SRC, k)]
    out_kv = [("general.architecture", "dflash"), ("general.name", str(kv.get("general.name", "")))]
    out_kv += [("dflash." + k[len(ARCH_SRC) + 1:], v) for k, v in kv.items() if k.startswith(ARCH_SRC + ".")]
    out_kv += [                                                           # то, что читает пакер
        ("dflash.block_count", g("block_count")),
        ("dflash.block_size", g("dflash.block_size")),
        ("dflash.expert_count", g("expert_count")),
        ("dflash.expert_used_count", g("expert_used_count")),
        ("dflash.target_layers", [int(x) for x in g("dflash.target_layer_ids")]),
    ]
    seen, uniq = set(), []
    for k, v in out_kv:
        if k not in seen:
            seen.add(k)
            uniq.append((k, v))
    out_kv = uniq

    out_t = []
    for name, typ, dims, off in tensors:
        if name in DROP:
            continue
        if name in HEAD:
            name = HEAD[name]
        elif not (name.startswith("blk.") and name.count(".") >= 2):
            raise SystemExit("%s: tensor %r is neither a blk. layer nor a known head" % (src, name))
        if typ not in TYPES:
            raise SystemExit("%s: %s has ggml type %d, unknown to the pack" % (src, name, typ))
        out_t.append((name, typ, dims, off))

    align = 32
    with open(dst, "wb") as f:
        f.write(b"GGUF")
        f.write(struct.pack("<IQQ", 3, len(out_t), len(out_kv)))
        for k, v in out_kv:
            w_kv(f, k, v)
        infos, pos = [], 0
        for name, typ, dims, _ in out_t:
            size = nbytes(dims, typ)
            infos.append((name, dims, typ, pos, size))
            pos += (size + align - 1) // align * align
        for name, dims, typ, off, _ in infos:
            w_str(f, name)
            f.write(struct.pack("<I", len(dims)))
            f.write(struct.pack("<%dQ" % len(dims), *dims))
            f.write(struct.pack("<IQ", typ, off))
        f.write(b"\0" * ((-f.tell()) % align))
        assert f.tell() % align == 0
        with open(src, "rb") as s:
            for _, _, _, old_off, size in infos:
                s.seek(data_start + old_off)
                left = size
                while left:
                    b = s.read(min(left, 1 << 26))
                    if not b:
                        raise SystemExit("%s: tensor data ends early" % src)
                    f.write(b)
                    left -= len(b)
                f.write(b"\0" * ((-size) % align))
    print("%s -> %s: %d tensors (%d dropped), %d kv keys, %s GiB data"
          % (src.name, dst.name, len(out_t), len(tensors) - len(out_t), len(out_kv),
             sum(i[4] for i in infos) / 2 ** 30))


if __name__ == "__main__":
    main()
