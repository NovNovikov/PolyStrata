// metal/poly_tests/mm_bench.cpp - Mmq::mul's time on the shapes of a prompt batch of LFM2.5-24B-A2B (64 experts of
// 2,048 x 1,536, 4 a token): the dense projection, the experts' gate|up and down, for each of mm.metal's tiles and
// for poly_mul.  The dense one takes poly_dq and poly_gemm whatever the tile.  Not a test: run by hand.
//   poly_mm_bench [reps [tokens, default 1,024]]
#include "test_utils.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "mmq.hpp"
#include "../src/poly/quantized.hpp"
#include "ggml.h"
#include "strata/gpu/metal.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
namespace k=strata::kernels;
namespace p=strata::poly;
int main(int argc,char** argv){
    const int reps=argc>1?std::atoi(argv[1]):20;
    g::Stream s;ck(g::stream_create(&s));g::metal::set_optimize(true);
    struct Shape{const char* name;int type,ni,no,n;};
    const int tokens=argc>2?std::atoi(argv[2]):1024,used=4;
    for(Shape sh:{Shape{"dense Q3_K 2048->6144",11,2048,6144,1},Shape{"experts gate|up IQ3_XXS 2048->3072",18,2048,3072,64},
                  Shape{"experts down IQ4_XS 1536->2048",23,1536,2048,64}}){
        const int rows=sh.n==1?tokens:tokens*used;
        // the rows of each expert, from random routing, sorted by expert as the engine sorts them
        std::vector<int32_t> bounds(sh.n+1,0);
        if(sh.n>1){std::mt19937 rng(5);std::vector<int> cnt(sh.n,0);for(int i=0;i<rows;++i)cnt[rng()%sh.n]++;for(int e=0;e<sh.n;++e)bounds[e+1]=bounds[e]+cnt[e];}
        const int most=sh.n>1?[&]{int m=0;for(int e=0;e<sh.n;++e)m=std::max(m,bounds[e+1]-bounds[e]);return m;}():rows;
        auto t=ggml_type(sh.type);ggml_quantize_init(t);
        const size_t per=k::native_mmvq_weight_bytes(sh.type,sh.ni,sh.no);
        std::vector<float> w(size_t(sh.no)*sh.ni);for(size_t i=0;i<w.size();++i)w[i]=std::sin(i*.37f);
        std::vector<unsigned char> one(per);std::vector<float> imp(sh.ni,1);ggml_quantize_chunk(t,w.data(),one.data(),0,sh.no,sh.ni,imp.data());
        Buffer dw(per*sh.n);for(int e=0;e<sh.n;++e)ck(g::memcpy_sync(static_cast<unsigned char*>(dw.ptr)+per*e,one.data(),per,g::kHostToDevice));
        Buffer dx(values(size_t(rows)*sh.ni)),q(p::Mmq::q8_bytes(rows,sh.ni)),y(size_t(rows)*sh.no*4),db(bounds.size()*4);
        db.put(bounds.data(),bounds.size()*4);
        const double flop=2.0*rows*sh.ni*sh.no;
        // the variants in turn, five rounds, each variant's best round: the Mac may be shared
        std::vector<const char*> tiles={nullptr,"32,64,64,4","64,64,32,4","128,64,64,4"};
        std::vector<double> best(tiles.size()+1,1e30);
        for(int round=0;round<5;++round) for(size_t v=0;v<=tiles.size();++v) {
            const bool units=v<tiles.size();
            p::metal_detail::allow_mm_rows(units);p::metal_detail::set_mm_tile(units?tiles[v]:nullptr);
            p::Mmq mmq;p::Mmq::quantize(dx.as(),nullptr,q.ptr,sh.type,sh.ni,sh.ni,rows,s);
            auto run=[&]{mmq.mul(dw.ptr,sh.type,sh.no,sh.ni,per,sh.n,q.ptr,sh.n>1?db.as<int32_t>():nullptr,nullptr,rows,most,y.as(),sh.no,s);};
            run();ck(g::stream_synchronize(s));
            const auto t0=std::chrono::steady_clock::now();
            for(int r=0;r<reps;++r)run();
            ck(g::stream_synchronize(s));
            best[v]=std::min(best[v],std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-t0).count()/reps);
        }
        for(size_t v=0;v<=tiles.size();++v)
            std::printf("%-40s %-14s %9.1f us  %6.2f TFLOP/s\n",sh.name,v==tiles.size()?"poly_mul":tiles[v]?tiles[v]:"default",
                        best[v],flop/best[v]*1e-6);
    }
    p::metal_detail::allow_mm_rows(true);p::metal_detail::set_mm_tile(nullptr);
    ck(g::stream_destroy(s));ggml_quantize_free();return 0;
}
