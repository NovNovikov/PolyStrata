// Dumps a GGUF's metadata KV pairs.  Hand-run: gguf_meta_dump FILE.gguf
#include "strata/artifact/gguf_reader.hpp"
#include <cstdio>
#include <string>

using strata::GgufFile;
using strata::MetaValue;
using strata::MetaType;

static std::string shown(const MetaValue& v) {
    switch (v.type) {
    case MetaType::STRING:
        return v.s;
    case MetaType::ARRAY:
        if (v.elem == MetaType::STRING) return "<" + std::to_string(v.count) + " strings>";
        return "<" + std::to_string(v.count) + " numbers>";
    default:
        if (v.type == MetaType::BOOL) return v.u ? "true" : "false";
        if (v.type == MetaType::F32 || v.type == MetaType::F64) {
            char out[32];
            std::snprintf(out, sizeof out, "%g", v.f);
            return out;
        }
        return std::to_string((long long) v.u);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    try {
        GgufFile g(argv[1]);
        for (const auto& [k, v] : g.metadata())
            std::printf("%s = %s\n", k.c_str(), shown(v).c_str());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "gguf_meta_dump: %s\n", e.what());
        return 1;
    }
}
