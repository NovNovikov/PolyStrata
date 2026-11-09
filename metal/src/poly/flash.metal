// attend_batch(): a prompt batch's attention on the matrix units, after metal/kernels/qsa_prompt.metal's qsa_flash.
// A threadgroup takes QB of the call's tokens for one query head and walks the keys they attend KC at a time with
// an online softmax.  SIMD group g owns head values 64 g .. 64 g + 63: its share of S = Q K^T, summed over the
// groups through threadgroup memory, and those 64 values of the output, P V.  The softmax takes a row a lane.
// Q is rounded to fp16 in threadgroup memory; K and V (fp16, or q8 codes times their scale) are decoded straight
// into the right-input cooperative tensors, zero past the keys.  P is fp16; the sums are fp32.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
struct Args {   // attention.metal's
    device void* p[10]; long pos;
    int dim,heads,kv,rows,window,ring,whole,stride,rot,sections,kind;
    float scale,eps;
};
constant constexpr int QB=16,KC=64;   // tokens a threadgroup, keys a step
inline half cache_half(device void* raw,device void* codes,device void* scales,uint at) {
    return codes?half(float(static_cast<device char*>(codes)[at])*float(static_cast<device half*>(scales)[at/32]))
                :static_cast<device half*>(raw)[at];
}
template<int D>
kernel void poly_flash(constant Args& a [[buffer(0)]],uint2 tg [[threadgroup_position_in_grid]],
                       ushort tid [[thread_index_in_threadgroup]],ushort sg [[simdgroup_index_in_threadgroup]],
                       ushort lane [[thread_index_in_simdgroup]]) {
    constexpr int NSG=D/64,NTH=NSG*32;
    threadgroup half qs[QB*D];          // the tokens' queries: [token][value]
    threadgroup float sp[NSG][QB*KC];   // each group's share of S: [token][key]
    threadgroup half p[QB*KC];          // the weights
    threadgroup uint koff[2][KC];       // a step's keys: where their head's row starts in the cache (by step parity)
    threadgroup float fac[QB],lsum[QB],mrun[QB];
    const int h=int(tg.x),t0=int(tg.y)*QB,kh=h/(a.heads/a.kv),n=min(QB,a.rows-t0);
    const long start=a.pos,last=a.whole?start+a.rows-1:start+t0+n-1;
    const long first=a.window>0?max(0L,start+t0-a.window+1):0;
    device const float* q=static_cast<device const float*>(a.p[0]);
    for(int i=tid;i<QB*D;i+=NTH) {
        const int r=i/D;
        qs[i]=r<n?half(q[(ulong(t0+r)*a.heads+h)*D+i%D]):half(0.0h);
    }
    if(tid<QB) { mrun[tid]=-INFINITY;lsum[tid]=0.0f; }
    constexpr auto dS=matmul2d_descriptor(QB,KC,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    constexpr auto dO=matmul2d_descriptor(QB,64,32,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<dS,execution_simdgroup> opS;
    matmul2d<dO,execution_simdgroup> opO;
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> tQ(qs,dextents<int32_t,2>(D,QB));
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> tP(p,dextents<int32_t,2>(KC,QB));
    auto rv=opO.template get_right_input_cooperative_tensor<half,half,float>();
    auto tPh=tP.template slice<32,QB>(0,0);
    using PType=__tensor_ops_detail::__remove_addrspace_t<decltype(tPh)>;
    using VType=__tensor_ops_detail::__remove_addrspace_t<decltype(rv)>;
    // tot: the running output; acc: one step's, fresh each step (qsa_flash: a run does not accumulate into elements
    // rescaled in place on the M4)
    auto acc=opO.template get_destination_cooperative_tensor<PType,VType,float>();
    auto tot=opO.template get_destination_cooperative_tensor<PType,VType,float>();
    for(uint16_t i=0;i<tot.get_capacity();++i) if(tot.is_valid_element(i)) tot[i]=0.0f;
    device void* kraw=a.p[1];device void* vraw=a.p[2];
    device void* kcodes=a.p[5];device void* kscales=a.p[6];device void* vcodes=a.p[7];device void* vscales=a.p[8];
    const int d0=sg*64;
    int step=0;
    for(long c0=first;c0<=last;c0+=KC,++step) {
        const int nc=int(min(long(KC),last-c0+1)),par=step&1;
        // the step before last read koff[par]; the barrier of the last step parts them
        for(int c=tid;c<KC;c+=NTH) {
            const long at=c0+c;
            koff[par][c]=c<nc?uint((ulong(a.ring?at%a.ring:at)*a.kv+kh)*D):0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            auto rk=opS.template get_right_input_cooperative_tensor<half,half,float>();
            auto tA=tQ.template slice<32,QB>(d0,0);
            using AType=__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>;
            using KType=__tensor_ops_detail::__remove_addrspace_t<decltype(rk)>;
            auto s=opS.template get_destination_cooperative_tensor<AType,KType,float>();
            for(uint16_t i=0;i<s.get_capacity();++i) if(s.is_valid_element(i)) s[i]=0.0f;
            for(int k0=d0;k0<d0+64;k0+=32) {
#pragma clang loop unroll(full)
                for(uint16_t i=0;i<rk.get_capacity();++i) {
                    const auto ix=rk.get_multidimensional_index(i);   // (value, key)
                    const int c=ix[1];
                    rk[i]=c<nc?cache_half(kraw,kcodes,kscales,koff[par][c]+uint(k0+ix[0])):half(0.0h);
                }
                tA=tQ.template slice<32,QB>(k0,0);
                opS.run(tA,rk,s);
            }
            tensor<threadgroup float,dextents<int32_t,2>,tensor_inline> tS(sp[sg],dextents<int32_t,2>(KC,QB));
            s.store(tS);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // the softmax, a token a lane of group 0: only its own keys (to itself, or the call's last with whole; within
        // the window) weigh
        if(sg==0 && lane<QB) {
            const int r=lane;
            if(r<n) {
                const long own=start+t0+r,lo=a.window>0?max(0L,own-a.window+1):0,hi=a.whole?start+a.rows-1:own;
                const int c1=int(clamp(lo-c0,0L,long(nc))),c2=int(clamp(hi-c0+1,0L,long(nc)));
                float m=mrun[r];
                for(int c=c1;c<c2;++c) {
                    float v=0.0f;
                    for(int g=0;g<NSG;++g) v+=sp[g][r*KC+c];
                    sp[0][r*KC+c]=v*a.scale;
                    m=max(m,v*a.scale);
                }
                const float f=m==-INFINITY?1.0f:exp(mrun[r]-m);
                float l=lsum[r]*f;
                for(int c=0;c<KC;++c) {
                    const float e=c>=c1 && c<c2?exp(sp[0][r*KC+c]-m):0.0f;
                    l+=e;
                    p[r*KC+c]=half(e);
                }
                mrun[r]=m;lsum[r]=l;fac[r]=f;
            } else {
                for(int c=0;c<KC;++c) p[r*KC+c]=half(0.0h);
                fac[r]=0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(uint16_t i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) acc[i]=0.0f;
        for(int k0=0;k0<KC;k0+=32) {
#pragma clang loop unroll(full)
            for(uint16_t i=0;i<rv.get_capacity();++i) {
                const auto ix=rv.get_multidimensional_index(i);   // (key, value)
                const int c=k0+ix[0];
                rv[i]=c<nc?cache_half(vraw,vcodes,vscales,koff[par][c]+uint(d0+ix[1])):half(0.0h);
            }
            tPh=tP.template slice<32,QB>(k0,0);
            opO.run(tPh,rv,acc);
        }
        for(uint16_t i=0;i<acc.get_capacity();++i)
            if(acc.is_valid_element(i)) tot[i]=fma(tot[i],fac[acc.get_multidimensional_index(i)[1]],acc[i]);
    }
    device float* out=static_cast<device float*>(a.p[3]);
    for(uint16_t i=0;i<tot.get_capacity();++i)
        if(tot.is_valid_element(i)) {
            const auto ix=tot.get_multidimensional_index(i);   // (value, token)
            const int r=ix[1];
            if(r<n) out[(ulong(t0+r)*a.heads+h)*D+d0+ix[0]]=tot[i]/lsum[r];
        }
}
typedef decltype(poly_flash<64>) poly_flash_t;
template [[host_name("poly_flash_64")]] kernel poly_flash_t poly_flash<64>;
template [[host_name("poly_flash_128")]] kernel poly_flash_t poly_flash<128>;
template [[host_name("poly_flash_256")]] kernel poly_flash_t poly_flash<256>;
