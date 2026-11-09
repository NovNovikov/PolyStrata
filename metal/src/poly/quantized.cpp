// PolyStrata projections using the pinned ggml Metal block decoders. All dispatch
// and graph ownership stays in PolyStrata's Metal runtime.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "launch.hpp"
#include "quantized.hpp"
#include "mmq.hpp"
#include "ggml.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
namespace strata::poly::metal_detail {
namespace {
std::string without(std::string s,const std::string& line) {
    auto p=s.find(line); if(p!=std::string::npos) s.erase(p,line.size());return s;
}
// ggml's block structs and Metal block decoders, ahead of a kernel file
std::string ggml_source() {
    const std::string root=STRATA_POLY_GGML_SOURCE;
    auto common=read_source((root+"/ggml-metal/kernels/common.h").c_str());
    common=without(common,"#include \"ggml-metal-impl.h\"");
    auto deq=read_source((root+"/ggml-metal/kernels/dequantize.h").c_str());
    deq=without(deq,"#include \"common.h\"");
    auto pos=deq.find("#include \"ggml-common.h\"");
    deq.replace(pos,std::string("#include \"ggml-common.h\"").size(),read_source((root+"/ggml-common.h").c_str()));
    return read_source((root+"/ggml-metal/ggml-metal-impl.h").c_str())+common+deq;
}
std::string quant_source() { return ggml_source()+read_source(STRATA_POLY_QUANT_SOURCE); }
const std::string& mm_source() {
    static const std::string source=ggml_source()+read_source(STRATA_POLY_MM_SOURCE);
    return source;
}
// The matrix-unit library compiles here (Metal 4's matmul2d); asked once
bool mm_ready() {
    static const bool ready=[] {
        std::string error;
        return gpu::metal::pipeline("poly_mm",mm_source().c_str(),"poly_mm_q4_K_64_64_64_4",error)!=nullptr;
    }();
    return ready;
}
struct MmTile { int tm=64,tn=64,tk=64,sg=4; };   // mm.metal's tile: rows, outputs, values a step, SIMD groups
struct MArgs {   // mm.metal's MmArgs
    const void* p[5]{};
    uint64_t stride=0,row_bytes=0,ld=0;
    int ni=0,no=0,total=0;
};
static_assert(sizeof(MArgs)==80);
struct GArgs {   // mm.metal's GemmArgs
    const void* p[3]{};
    int m=0,n=0,k=0,pad=0;
};
static_assert(sizeof(GArgs)==40);
// The fp16 copy of a dense matrix (poly_dq) that poly_gemm reads: grown to the largest one, kept.  Larger matrices
// than kDqMax bytes of halves (an output head) and calls of fewer than kDqRows rows stay with poly_mm.
void* dq_buf=nullptr;
size_t dq_bytes=0;
constexpr size_t kDqMax=size_t(64)<<20;
constexpr int64_t kDqRows=128;
}
namespace {
bool mm_allowed=!(std::getenv("STRATA_METAL_MM") && std::string(std::getenv("STRATA_METAL_MM"))=="0");
MmTile mm_tile;
}
void allow_mm_rows(bool on) { mm_allowed=on; }
void set_mm_tile(const char* tile) {
    mm_tile=MmTile{};
    if(tile) std::sscanf(tile,"%d,%d,%d,%d",&mm_tile.tm,&mm_tile.tn,&mm_tile.tk,&mm_tile.sg);
}
bool mm_rows(int64_t cols) { return mm_allowed && cols>0 && cols%mm_tile.tk==0 && gpu::metal::optimize() && mm_ready(); }
bool mm_gemm(const void* a,const void* b,float* y,int m,int n,int k,bool f32,void* s) {
    if(!mm_allowed || m<=0 || n<=0 || k<=0 || !gpu::metal::optimize() || !mm_ready()) return false;
    GArgs g;g.p[0]=a;g.p[1]=b;g.p[2]=y;g.m=m;g.n=n;g.k=k;
    const int tn=n<=32?32:64;
    const std::string name=std::string("poly_gemm_")+(f32?"float":"half")+"_64_"+std::to_string(tn)+"_4";
    dispatch("poly_mm",mm_source(),name.c_str(),g,s,{uint32_t((n+tn-1)/tn),uint32_t((m+63)/64),1},128);
    return true;
}
void qrun(const std::string& name,const QArgs& a,void* s,gpu::metal::Grid groups) {
    static const auto source=quant_source();
    const bool indirect = name.starts_with("poly_mul_") && (a.kind==1 || a.kind==4 || a.kind==5);
    dispatch("poly_quantized",source,name.c_str(),a,s,groups,32,{a.p,a.kind==8?size_t(3*a.count):6,true,indirect?a.p[0]:nullptr});
}
const char* quant_name(int t) {
    switch(t) {
#define Q(id,name) case id:return #name;
    Q(2,q4_0) Q(3,q4_1) Q(6,q5_0) Q(7,q5_1) Q(8,q8_0) Q(10,q2_K) Q(11,q3_K) Q(12,q4_K) Q(13,q5_K)
    Q(14,q6_K) Q(16,iq2_xxs) Q(17,iq2_xs) Q(18,iq3_xxs) Q(19,iq1_s) Q(20,iq4_nl) Q(21,iq3_s)
    Q(22,iq2_s) Q(23,iq4_xs) Q(29,iq1_m) Q(39,mxfp4) Q(42,q2_0)
#undef Q
    default:return nullptr;
    }
}
}
namespace strata::kernels {
using namespace poly::metal_detail;
bool native_mmvq_supported(int t) noexcept { return quant_name(t); }
bool native_mmvq_ids_supported(int t) { return native_mmvq_supported(t); }
size_t native_q8_1_bytes(int n,int rows) {
    if(n<=0 || n%32 || rows<=0) throw std::invalid_argument("Q8_1 shape");
    return size_t(n/32)*36*rows;
}
size_t native_mmvq_weight_bytes(int t,int ni,int no) {
    if(!native_mmvq_supported(t) || ni<=0 || no<=0 || ni%ggml_blck_size(ggml_type(t))) throw std::invalid_argument("Metal quantized matrix shape");
    return ggml_row_size(ggml_type(t),ni)*no;
}
void native_quantize_q8_1(const float* x,void* q,int n,int rows,void* s) {
    native_q8_1_bytes(n,rows);
    QArgs a;a.p[0]=x;a.p[1]=q;a.ni=n;a.rows=rows;a.xstride=n;
    qrun("poly_quantize",a,s,{uint32_t(n/32),uint32_t(rows),1});
}
void native_add_rms_norm_q8_1(float* x,const float* add,const float* w,float* y,void* q,int n,int rows,float eps,int,void* s) {
    if(add) poly::axpy(add,1,x,n*rows,true,s);
    poly::rms_norm(x,w,y,n,rows,eps,s);
    native_quantize_q8_1(y,q,n,rows,s);
}
void native_mmvq(int type,const void* w,const void* x,float* y,int ni,int no,int rows,void* s) {
    native_mmvq_weight_bytes(type,ni,no);
    QArgs a;a.p[0]=w;a.p[1]=x;a.p[2]=y;a.ni=ni;a.no=no;a.rows=rows;a.xstride=native_q8_1_bytes(ni);a.ystride=no;
    qrun(std::string("poly_mul_")+quant_name(type),a,s,{uint32_t(no),uint32_t(rows),1});
}
bool native_mmvq_slots(int t,const MmvqSlots& a,int count,int ni,int no,void* s) {
    if(!native_mmvq_supported(t) || count<0 || count>8) return false;
    if(!count) return true;
    if(!gpu::metal::optimize()) {
        for(int i=0;i<count;++i) native_mmvq(t,a.w[i],a.x[i],a.y[i],ni,no,1,s);
        return true;
    }
    native_mmvq_weight_bytes(t,ni,no);
    QArgs q; q.ni=ni; q.no=no; q.rows=count; q.count=count; q.ystride=1; q.kind=8;
    for(int i=0;i<count;++i) { q.p[3*i]=a.w[i]; q.p[3*i+1]=a.x[i]; q.p[3*i+2]=a.y[i]; }
    qrun(std::string("poly_mul_")+quant_name(t),q,s,{uint32_t(no),uint32_t(count),1});
    return true;
}
bool native_mmvq_ids(int t,const MmvqIds& a,int count,int ni,int no,void* s) {
    if(!native_mmvq_supported(t)) return false;
    native_mmvq_weight_bytes(t,ni,no);
    QArgs q;q.p[0]=a.table;q.p[1]=a.x;q.p[2]=a.y;q.p[3]=a.ids;q.ni=ni;q.no=no;q.rows=count;
    q.xstride=a.x_stride;q.ystride=a.y_stride;q.offset=a.offset;q.every=a.x_every;q.kind=1;
    qrun(std::string("poly_mul_")+quant_name(t),q,s,{uint32_t(no),uint32_t(count),1});return true;
}
}
namespace strata::poly {
using namespace metal_detail;
Mmq::Mmq() = default;
Mmq::~Mmq() = default;
bool Mmq::fits(int t,int64_t rows,int64_t cols) {
    return kernels::native_mmvq_supported(t) && rows>0 && rows<=INT32_MAX && cols>0 && cols<=INT32_MAX && cols%ggml_blck_size(ggml_type(t))==0;
}
// room for either form of the rows: q8_1, or fp16 on the matrix-unit path (mm_rows())
size_t Mmq::q8_bytes(int64_t rows,int64_t cols) {
    return std::max(kernels::native_q8_1_bytes(int(cols),int(rows)),size_t(rows)*size_t(cols)*2);
}
void Mmq::quantize(const float* x,const int32_t* ids,void* q,int,int64_t cols,int64_t ld,int64_t rows,void* s) {
    QArgs a;a.p[0]=x;a.p[1]=q;a.p[2]=ids;a.ni=int(cols);a.rows=int(rows);a.xstride=ld;
    if(mm_rows(cols)) {
        static const auto source=quant_source();
        const uint32_t threads=uint32_t(std::min<int64_t>(256,cols/4));
        dispatch("poly_quantized",source,"poly_tohalf",a,s,{uint32_t((cols/4+threads-1)/threads),uint32_t(rows),1},threads,
                 {a.p,3,true});
        return;
    }
    qrun("poly_quantize",a,s,{uint32_t(cols/32),uint32_t(rows),1});
}
void Mmq::mul(const void* w,int type,int64_t nr,int64_t nc,size_t stride,int n,const void* x,
              const int32_t* bounds,const int32_t* ids,int64_t total,int64_t max_rows,float* y,int64_t ld,void* s) {
    if(n<=0 || total<=0 || max_rows<=0) return;
    if(!fits(type,nr,nc)) throw std::invalid_argument("Metal MMQ shape");
    const size_t half_bytes=size_t(nr)*size_t(nc)*2;
    if(mm_rows(nc) && !bounds && !ids && ld==nr && total>=kDqRows && half_bytes<=kDqMax) {
        if(half_bytes>dq_bytes) {
            if(dq_buf) { gpu::stream_synchronize(static_cast<gpu::Stream>(s)); gpu::free_device(dq_buf); }
            dq_buf=nullptr;dq_bytes=0;
            if(gpu::malloc_device_raw(&dq_buf,half_bytes)==gpu::kSuccess) dq_bytes=half_bytes;
            else dq_buf=nullptr;
        }
        if(dq_buf) {
            MArgs m;m.p[0]=w;m.p[1]=dq_buf;m.row_bytes=ggml_row_size(ggml_type(type),nc);m.ni=int(nc);m.no=int(nr);
            const uint64_t threads=uint64_t(nr)*uint64_t(nc/16);
            dispatch("poly_mm",mm_source(),(std::string("poly_dq_")+quant_name(type)).c_str(),m,s,
                     {uint32_t((threads+255)/256),1,1},256);
            mm_gemm(x,dq_buf,y,int(total),int(nr),int(nc),false,s);
            return;
        }
    }
    if(mm_rows(nc)) {   // TM rows of one matrix and TN outputs a threadgroup, the rows past a matrix's bounds idle
        MArgs m;m.p[0]=w;m.p[1]=x;m.p[2]=y;m.p[3]=ids;m.p[4]=bounds;
        m.stride=stride;m.row_bytes=ggml_row_size(ggml_type(type),nc);m.ld=uint64_t(ld);m.ni=int(nc);m.no=int(nr);m.total=int(total);
        const int64_t rows=bounds?max_rows:total;
        const MmTile& t=mm_tile;
        const std::string name=std::string("poly_mm_")+quant_name(type)+"_"+std::to_string(t.tm)+"_"+std::to_string(t.tn)+"_"+
                               std::to_string(t.tk)+"_"+std::to_string(t.sg);
        dispatch("poly_mm",mm_source(),name.c_str(),m,s,
                 {uint32_t((nr+t.tn-1)/t.tn),uint32_t((rows+t.tm-1)/t.tm),uint32_t(bounds?n:1)},uint32_t(32*t.sg));
        return;
    }
    const bool opt=gpu::metal::optimize(), flat=opt && bounds && total<max_rows*n;
    QArgs a;a.p[0]=w;a.p[1]=x;a.p[2]=y;a.p[3]=ids;a.p[4]=bounds;a.ni=int(nc);a.no=int(nr);a.rows=int(total);a.count=n;
    a.xstride=kernels::native_q8_1_bytes(int(nc));a.ystride=ld;a.stride=stride;a.kind=flat?7:(opt?6:2);
    qrun(std::string("poly_mul_")+quant_name(type),a,s,{uint32_t(nr),uint32_t(flat?total:(opt?max_rows:total)),uint32_t(flat?1:n)});
}
}
