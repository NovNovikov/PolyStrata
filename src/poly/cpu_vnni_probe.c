// Test-only replacement of ggml's x86 dot products. The production ggml-cpu library stays unchanged.
#define __AVXVNNI__ 1
#include "../../third_party/llama.cpp/ggml/src/ggml-cpu/arch/x86/quants.c"
