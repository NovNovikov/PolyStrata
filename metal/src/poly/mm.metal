// A prompt batch's projections (Mmq::mul) on the matrix units: Metal 4's MetalPerformancePrimitives matmul2d (on the
// M5, its neural accelerators), as the routed experts of metal/kernels/moe_mm.metal do for the Strata engine.  A
// threadgroup takes TM rows of one matrix (an expert's rows lie between its bounds) and TN of its outputs.  Per TK
// values of a row it decodes the TN x TK weights with ggml's decoders into threadgroup memory as fp16, and matmul2d
// multiplies the rows' fp16 activations, read where they lie, by them, summing in fp32 in its destination; the
// sums go out from there, row r to row ids[r] of y.  ggml's pinned dequantize.h comes first (quantized.cpp).
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;

struct MmArgs {
    device void* p[5];   // the first matrix, the rows' fp16 activations, y, ids (or null), bounds (or null)
    ulong stride;        // bytes from one matrix to the next
    ulong row_bytes;     // bytes of one matrix row
    ulong ld;            // floats from one row of y to the next
    int ni, no, total;   // a row's values, the matrices' rows, the call's rows
};

template<typename Block,int QK,void Deq(device const Block*,short,thread half4x4&),int TM,int TN,int TK,int SG>
kernel void poly_mm(constant MmArgs& a [[buffer(0)]],uint3 tg [[threadgroup_position_in_grid]],
                    ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int NTH=SG*32;
    threadgroup half wb[TN*TK];
    device const int* bounds=static_cast<device const int*>(a.p[4]);
    device const int* ids=static_cast<device const int*>(a.p[3]);
    const int e=int(tg.z),lo=bounds?bounds[e]:0,hi=bounds?bounds[e+1]:a.total,r0=lo+int(tg.y)*TM;
    if(r0>=hi) return;   // past this matrix's rows: the whole threadgroup leaves
    const int f0=int(tg.x)*TN;
    device const uchar* w0=static_cast<device const uchar*>(a.p[0])+ulong(e)*a.stride+ulong(f0)*a.row_bytes;
    // the matrix's rows: a tile past them reads nothing there
    tensor<device half,dextents<int32_t,2>,tensor_inline> tX(static_cast<device half*>(a.p[1])+ulong(lo)*a.ni,
                                                             dextents<int32_t,2>(a.ni,hi-lo));
    tensor<threadgroup half,dextents<int32_t,2>,tensor_inline> tW(wb,dextents<int32_t,2>(TK,TN));
    constexpr auto desc=matmul2d_descriptor(TM,TN,TK,false,true,false,matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto mA0=tX.slice(0,r0-lo);
    auto acc=op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(mA0)>,
                                                            __tensor_ops_detail::__remove_addrspace_t<decltype(tW)>,float>();
    for(uint16_t i=0;i<acc.get_capacity();++i) if(acc.is_valid_element(i)) acc[i]=0.0f;
    for(int k0=0;k0<a.ni;k0+=TK) {
        for(int it=tid;it<TN*(TK/16);it+=NTH) {
            const int n=it/(TK/16),k=k0+(it%(TK/16))*16;
            half4x4 v=half4x4(0.0h);
            if(f0+n<a.no) Deq(reinterpret_cast<device const Block*>(w0+ulong(n)*a.row_bytes)+k/QK,short((k%QK)/16),v);
            threadgroup half4* dst=reinterpret_cast<threadgroup half4*>(wb+n*TK+(it%(TK/16))*16);
            dst[0]=v[0];dst[1]=v[1];dst[2]=v[2];dst[3]=v[3];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto mA=tX.slice(k0,r0-lo);
        op.run(mA,tW,acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float* y=static_cast<device float*>(a.p[2]);
    for(uint16_t i=0;i<acc.get_capacity();++i)
        if(acc.is_valid_element(i)) {
            const auto ix=acc.get_multidimensional_index(i);   // (output, row)
            const int r=r0+ix[1],f=f0+ix[0];
            if(r<hi && f<a.no) y[ulong(ids?ids[r]:r)*a.ld+f]=acc[i];
        }
}

// A dense matrix's product decodes it once more for each TM rows in poly_mm; instead (dense_mm.metal's dq_f16 and
// gemm_f16): poly_dq writes the matrix in fp16 (p[0] -> p[1], a thread to each 16 values), and poly_gemm multiplies
// rows by it with the whole K in one matmul2d, both operands where they lie.
template<typename Block,int QK,void Deq(device const Block*,short,thread half4x4&)>
kernel void poly_dq(constant MmArgs& a [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    const uint per=uint(a.ni)/16,r=i/per,k=(i%per)*16;
    if(r>=uint(a.no)) return;
    half4x4 v;
    Deq(reinterpret_cast<device const Block*>(static_cast<device const uchar*>(a.p[0])+ulong(r)*a.row_bytes)+k/QK,
        short((k%QK)/16),v);
    device half4* dst=reinterpret_cast<device half4*>(static_cast<device half*>(a.p[1])+ulong(r)*a.ni+k);
    dst[0]=v[0];dst[1]=v[1];dst[2]=v[2];dst[3]=v[3];
}
struct GemmArgs { device void* p[3]; int m,n,k,pad; };   // A (m rows), B (n rows), Y (m rows of n floats); k values
template<typename T,int TM,int TN,int SG>
kernel void poly_gemm(constant GemmArgs& a [[buffer(0)]],uint3 tg [[threadgroup_position_in_grid]]) {
    tensor<device T,dextents<int32_t,2>,tensor_inline> tA(static_cast<device T*>(a.p[0]),dextents<int32_t,2>(a.k,a.m));
    tensor<device T,dextents<int32_t,2>,tensor_inline> tB(static_cast<device T*>(a.p[1]),dextents<int32_t,2>(a.k,a.n));
    tensor<device float,dextents<int32_t,2>,tensor_inline> tY(static_cast<device float*>(a.p[2]),
                                                              dextents<int32_t,2>(a.n,a.m));
    constexpr auto desc=matmul2d_descriptor(TM,TN,dynamic_length_v<int>,false,true,false);
    matmul2d<desc,execution_simdgroups<SG>> op;
    auto mA=tA.slice(0,int(tg.y)*TM);
    auto mB=tB.slice(0,int(tg.x)*TN);
    auto mY=tY.slice(int(tg.x)*TN,int(tg.y)*TM);   // rows and outputs past m and n are not written
    op.run(mA,mB,mY);
}
#define GEMM(T,TM,TN,SG) template [[host_name("poly_gemm_" #T "_" #TM "_" #TN "_" #SG)]] kernel decltype(poly_gemm<T,TM,TN,SG>) poly_gemm<T,TM,TN,SG>;
GEMM(half,64,64,4) GEMM(half,64,32,4) GEMM(float,64,64,4) GEMM(float,64,32,4)

// poly_mm_<type>_<TM>_<TN>_<TK>_<SG>: TM rows of a matrix, TN outputs, TK values a step, SG SIMD groups.  Every type
// has the default tile; the others are for poly_mm_bench, on the expert types of LFM2.5 and Qwen3.6.
#define MM_(name,QK,TM,TN,TK,SG) template [[host_name("poly_mm_" #name "_" #TM "_" #TN "_" #TK "_" #SG)]] kernel decltype(poly_mm<block_##name,QK,dequantize_##name<half4x4>,TM,TN,TK,SG>) poly_mm<block_##name,QK,dequantize_##name<half4x4>,TM,TN,TK,SG>;
#define MM(name,QK) MM_(name,QK,64,64,64,4) template [[host_name("poly_dq_" #name)]] kernel decltype(poly_dq<block_##name,QK,dequantize_##name<half4x4>>) poly_dq<block_##name,QK,dequantize_##name<half4x4>>;
#define MM_TRY(name,QK) MM_(name,QK,32,64,64,4) MM_(name,QK,64,64,32,4) MM_(name,QK,128,64,64,4)
MM(q4_0,32) MM(q4_1,32) MM(q5_0,32) MM(q5_1,32) MM(q8_0,32) MM(q2_0,64) MM(mxfp4,32)
MM(q2_K,256) MM(q3_K,256) MM(q4_K,256) MM(q5_K,256) MM(q6_K,256)
MM(iq2_xxs,256) MM(iq2_xs,256) MM(iq3_xxs,256) MM(iq1_s,256) MM(iq4_nl,32)
MM(iq3_s,256) MM(iq2_s,256) MM(iq4_xs,256) MM(iq1_m,256)
MM_TRY(iq3_xxs,256) MM_TRY(iq4_xs,256) MM_TRY(iq2_xs,256)
