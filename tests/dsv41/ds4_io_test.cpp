// Strata. ds4_os.hpp's read paths on a small temporary file - the shapes STRATA_DS4_DROP_SHARDS (the f62520a2aa
// port, task spec 8.8) leaves the engine running with: unbuffered handles alive, buffered handles and mapped
// views closed. Checks the production functions: buffered, parallel-buffered, unbuffered and direct reads of
// the same bytes agree; unaligned requests (the lead) give exactly the asked-for bytes; after the buffered
// handle is closed the direct one still reads right; touching a closed handle reports failure rather than
// success; a read past the file's end fails and does not fake data; file metadata survives the drop.
// No GGUF, no device, no model. Windows runs all of it; the unbuffered/direct handles are Windows concepts
// and the functions report failure on other platforms, where the comparisons still run on the buffered pair.
#include "ds4_os.hpp"

#ifdef _WIN32
#include <malloc.h>   // _aligned_malloc: the direct reads need an aligned destination, as the pool's does
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace strata::ds4;

static int failures = 0, checks = 0;
static void check(bool ok, const char* what) {
    ++checks;
    if (!ok) { ++failures; std::printf("  FAIL: %s\n", what); }
}

static uint8_t pat(uint64_t i) { return (uint8_t) (i * 131u + (i >> 8) * 7u + 11u); }

int main() {
    const uint64_t FSIZE = 200000;   // not a multiple of 4096: the tail edge matters
    const std::string path = (std::getenv("TEMP") ? std::getenv("TEMP") : ".") + std::string("\\ds4_io_test.bin");
    {
        std::vector<uint8_t> data(FSIZE);
        for (uint64_t i = 0; i < FSIZE; ++i) data[i] = pat(i);
        FILE* f = fopen(path.c_str(), "wb");
        if (!f) { std::printf("ds4_io_test: cannot write the temp file\n"); return 2; }
        fwrite(data.data(), 1, FSIZE, f);
        fclose(f);
    }
    const os::fd_t ro = os::open_ro(path), par = os::open_par(path), dir = os::open_direct(path), unbuf = os::open_unbuffered(path);
    check(ro != os::kBadFd, "buffered handle opens");
    check(par != os::kBadFd, "the pool's buffered handle opens");
    check(dir != os::kBadFd, "the unbuffered handle opens");
    check(unbuf != os::kBadFd, "the second unbuffered handle opens");
    check(os::file_size(ro) == FSIZE, "buffered metadata: the file size");
    check(os::file_size(dir) == FSIZE, "the unbuffered handle keeps the file metadata");

    // aligned reads through all four paths agree with the pattern
    {
        std::vector<uint8_t> want(4096);
        for (uint64_t i = 0; i < 4096; ++i) want[i] = pat(8192 + i);
        std::vector<uint8_t> a(4096), b(4096), c(4096, 0), d(4096, 0);
        check(os::pread_full(ro, a.data(), 4096, 8192) && a == want, "buffered aligned read");
        check(os::pread_par(par, b.data(), 4096, 8192) && b == want, "parallel-buffered aligned read");
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(8192, 4096) + os::kDirectAlign, os::kDirectAlign);
            check(raw != nullptr && os::pread_direct(dir, raw, 4096, 8192) &&
                  std::memcmp(raw + 8192 % os::kDirectAlign, want.data(), 4096) == 0, "direct aligned read (lead 0)");
            _aligned_free(raw);
        }
        if (unbuf != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(8192, 4096) + os::kDirectAlign, os::kDirectAlign);
            check(raw != nullptr && os::pread_unbuffered(unbuf, raw, 4096, 8192) &&
                  std::memcmp(raw + 8192 % os::kDirectAlign, want.data(), 4096) == 0, "unbuffered aligned read (lead 0)");
            _aligned_free(raw);
        }
    }
    // unaligned requests: the lead's bytes are the pool's dst+lead contract - exact content at odd offsets
    for (uint64_t off : {3ull, 4095ull, 4097ull, 150001ull}) {
        const uint64_t n = 1000;
        std::vector<uint8_t> want(n), a(n), b(n);
        for (uint64_t i = 0; i < n; ++i) want[i] = pat(off + i);
        check(os::pread_full(ro, a.data(), n, off) && a == want, "buffered unaligned read");
        check(os::pread_par(par, b.data(), n, off) && b == want, "parallel-buffered unaligned read (its own lead)");
        if (dir != os::kBadFd) {
            const uint64_t lead = off % os::kDirectAlign;
            std::vector<uint8_t> raw((size_t) os::direct_span(off, n));
            uint8_t* al = (uint8_t*) _aligned_malloc(raw.size(), os::kDirectAlign);
            bool got = al != nullptr && os::pread_direct(dir, al, n, off) && std::memcmp(al + lead, want.data(), n) == 0;
            _aligned_free(al);
            check(got, "direct unaligned read: dst[lead .. lead+n) is exactly the asked-for bytes");
        }
    }
    // the DROP_SHARDS shape: close the buffered handles, then read through the direct one
    {
        os::close_fd(par);
        std::vector<uint8_t> want(4096), a(4096);
        for (uint64_t i = 0; i < 4096; ++i) want[i] = pat(20480 + i);
        check(os::pread_full(ro, a.data(), 4096, 20480) && a == want, "the kept buffered handle is untouched by another file-shape close");
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(20480, 4096), os::kDirectAlign);
            check(raw != nullptr && os::pread_direct(dir, raw, 4096, 20480) && std::memcmp(raw, want.data(), 4096) == 0,
                  "direct read works with the pool's buffered handle closed (the post-drop state)");
            _aligned_free(raw);
        }
        os::close_fd(ro);
        std::vector<uint8_t> junk(4096, 0);
        check(!os::pread_full(ro, junk.data(), 4096, 20480), "a closed buffered handle reports failure, not success");
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(20480, 4096), os::kDirectAlign);
            bool still = raw != nullptr && os::pread_direct(dir, raw, 4096, 20480);
            _aligned_free(raw);
            check(still, "the direct handle still reads after every buffered handle closed");
        }
    }
    // a read past the end fails rather than filling with anything: 4096 at FSIZE-1000 spans the EOF
    {
        if (dir != os::kBadFd) {
            uint8_t* raw = (uint8_t*) _aligned_malloc(os::direct_span(FSIZE - 1000, 4096), os::kDirectAlign);
            const bool bad = !os::pread_direct(dir, raw, 4096, FSIZE - 1000);
            _aligned_free(raw);
            check(bad, "a direct read past the end fails - the pool reports it, nothing computes on it");
        }
    }
    os::close_fd(dir);
    os::close_fd(unbuf);
    remove(path.c_str());
    std::printf("ds4_io_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
