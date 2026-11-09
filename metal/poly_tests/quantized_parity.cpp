#include "test_utils.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "mmq.hpp"
#include "../src/poly/quantized.hpp"
#include "ggml.h"
#include "strata/gpu/metal.hpp"
#include <cstring>
namespace k=strata::kernels;
namespace p=strata::poly;
// The rows' values as Mmq::mul() saw them: q8_1 decoded, or fp16 on the matrix-unit path
std::vector<float> mmq_rows(Buffer& q,const std::vector<float>& x,int ni,int rows){
    std::vector<float> v(x.size());
    if(p::metal_detail::mm_rows(ni)){for(size_t i=0;i<x.size();++i)v[i]=float(_Float16(x[i]));return v;}
    auto qb=q.read<unsigned char>(k::native_q8_1_bytes(ni,rows));
    for(size_t i=0;i<x.size();++i){_Float16 d;std::memcpy(&d,qb.data()+(i/32)*36,2);v[i]=float(d)*reinterpret_cast<int8_t*>(qb.data())[(i/32)*36+4+i%32];}
    return v;
}
int main(){
    int bad=0;g::Stream stream;ck(g::stream_create(&stream));
    const int ni=256,no=5,rows=3,matrices=2;auto wf=values(matrices*no*ni),x=values(rows*ni,.8f);std::vector<float> importance(ni,1);
    Buffer dx(x),q(p::Mmq::q8_bytes(rows,ni)),y(rows*no*4);k::native_quantize_q8_1(dx.as(),q.ptr,ni,rows,stream);ck(g::stream_synchronize(stream));
    auto qb=q.read<unsigned char>(k::native_q8_1_bytes(ni,rows));std::vector<float> xq(x.size());
    for(size_t i=0;i<x.size();++i){_Float16 d;std::memcpy(&d,qb.data()+(i/32)*36,2);xq[i]=float(d)*reinterpret_cast<int8_t*>(qb.data())[(i/32)*36+4+i%32];}
    // (optimized, matrix units): plain, the compact dispatch of poly_mul, and mm.metal where matmul2d compiles
    for(auto [optimized,units]:{std::pair{false,false},{true,false},{true,true}}) { g::metal::set_optimize(optimized);p::metal_detail::allow_mm_rows(units);
    for(int type:{2,3,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,29,39,42}){
        auto t=ggml_type(type);ggml_quantize_init(t);size_t size=k::native_mmvq_weight_bytes(type,ni,no);
        std::vector<unsigned char> weights(size*matrices);
        ggml_quantize_chunk(t,wf.data(),weights.data(),0,matrices*no,ni,importance.data());
        auto traits=ggml_get_type_traits(t);std::vector<float> dwf(wf.size());traits->to_float(weights.data(),dwf.data(),dwf.size());
        Buffer dw(weights.size());dw.put(weights.data(),weights.size());
        // the products of rows of `in` (the inputs' values as read) with matrices experts[r], into rows outputs[r]
        auto product=[&](const std::vector<float>& in,bool half_weights,const std::vector<int>& experts,const std::vector<int>& inputs,const std::vector<int>& outputs){
            std::vector<float> ref(rows*no);
            for(int r=0;r<rows;++r)for(int o=0;o<no;++o){double sum=0;for(int i=0;i<ni;++i){float w=dwf[(experts[r]*no+o)*ni+i];sum+=double(half_weights?float(_Float16(w)):w)*in[inputs[r]*ni+i];}ref[outputs[r]*no+o]=sum;}return ref;
        };
        auto reference=[&](const std::vector<int>& experts,const std::vector<int>& inputs,const std::vector<int>& outputs){return product(xq,false,experts,inputs,outputs);};
        k::native_mmvq(type,dw.ptr,q.ptr,y.as(),ni,no,rows,stream);ck(g::stream_synchronize(stream));
        bad+=compare(y.read(rows*no),reference({0,0,0},{0,1,2},{0,1,2}),ggml_type_name(t),1e-4f);
        std::vector<const void*> table={dw.ptr,static_cast<unsigned char*>(dw.ptr)+size};std::vector<int32_t> ids={1,0,1};
        Buffer tab(table.size()*sizeof(void*)),di(ids.size()*4);tab.put(table.data(),table.size()*sizeof(void*));di.put(ids.data(),ids.size()*4);
        k::MmvqIds selected{tab.as<const void*>(),di.as<int32_t>(),0,q.ptr,k::native_q8_1_bytes(ni),1,y.as(),no};
        k::native_mmvq_ids(type,selected,rows,ni,no,stream);ck(g::stream_synchronize(stream));
        bad+=compare(y.read(rows*no),reference({1,0,1},{0,1,2},{0,1,2}),"expert ids",1e-4f);
        k::MmvqSlots slots{};
        for(int r=0;r<rows;++r){slots.w[r]=table[ids[r]];slots.x[r]=static_cast<unsigned char*>(q.ptr)+r*k::native_q8_1_bytes(ni);slots.y[r]=y.as()+r*no;}
        k::native_mmvq_slots(type,slots,rows,ni,no,stream);ck(g::stream_synchronize(stream));
        bad+=compare(y.read(rows*no),reference({1,0,1},{0,1,2},{0,1,2}),"expert slots",1e-4f);
        std::vector<int32_t> gather={2,0,1},bounds={0,1,3},scatter={1,2,0};
        Buffer dg(12),db(12),ds(12);dg.put(gather.data(),12);db.put(bounds.data(),12);ds.put(scatter.data(),12);
        p::Mmq mmq;p::Mmq::quantize(dx.as(),dg.as<int32_t>(),q.ptr,type,ni,ni,rows,stream);
        mmq.mul(dw.ptr,type,no,ni,size,matrices,q.ptr,db.as<int32_t>(),ds.as<int32_t>(),rows,2,y.as(),no,stream);ck(g::stream_synchronize(stream));
        // the gathered rows as Mmq::quantize() left them, in gather order
        std::vector<float> gx(x.size());for(int r=0;r<rows;++r)std::memcpy(&gx[r*ni],&x[gather[r]*ni],ni*4);
        const bool units_used=p::metal_detail::mm_rows(ni);
        bad+=compare(y.read(rows*no),product(mmq_rows(q,gx,ni,rows),units_used,{0,1,1},{0,1,2},{1,2,0}),
                     units_used?"batch gather/scatter, matrix units":"batch gather/scatter",1e-4f);
        // Restore the input ordering for the next weight format.
        k::native_quantize_q8_1(dx.as(),q.ptr,ni,rows,stream);
    }
    }
    // Empty first expert and nonzero bounds: the untouched row and columns stay as they were, the global gather and
    // scatter indices hold, and the plain and compact poly_mul agree to the bit.
    for(int type:{12,13,14}) {
        const int ni2=512,no2=7,rows2=5,stride=11;
        auto w=values(2*no2*ni2), x2=values(rows2*ni2);std::vector<float> imp(ni2,1);
        auto ty=ggml_type(type);ggml_quantize_init(ty);
        size_t wb=k::native_mmvq_weight_bytes(type,ni2,no2);
        std::vector<unsigned char> packed(wb*2);ggml_quantize_chunk(ty,w.data(),packed.data(),0,2*no2,ni2,imp.data());
        std::vector<float> dwf(w.size());ggml_get_type_traits(ty)->to_float(packed.data(),dwf.data(),dwf.size());
        Buffer dw(packed.size()),dx2(x2),qx(p::Mmq::q8_bytes(rows2,ni2)),out(rows2*stride*4),bd(12),id(20);
        dw.put(packed.data(),packed.size());int32_t bounds[]={1,1,5},ids[]={3,4,2,1,0};bd.put(bounds,12);id.put(ids,20);
        std::vector<float> old;
        for(auto [optimized,units]:{std::pair{false,false},{true,false},{true,true}}) {
            g::metal::set_optimize(optimized);p::metal_detail::allow_mm_rows(units);
            p::Mmq::quantize(dx2.as(),nullptr,qx.ptr,type,ni2,ni2,rows2,stream);ck(g::memset_async(out.ptr,0,rows2*stride*4,stream));
            p::Mmq mmq;mmq.mul(dw.ptr,type,no2,ni2,wb,2,qx.ptr,bd.as<int32_t>(),id.as<int32_t>(),rows2,4,out.as(),stride,stream);
            ck(g::stream_synchronize(stream));auto got=out.read(rows2*stride);
            const bool units_used=p::metal_detail::mm_rows(ni2);
            auto in=mmq_rows(qx,x2,ni2,rows2);std::vector<float> ref(rows2*stride,0);
            for(int r=bounds[0];r<bounds[2];++r)for(int o=0;o<no2;++o){
                const int e=r<bounds[1]?0:1;double sum=0;
                for(int i=0;i<ni2;++i){float v=dwf[(e*no2+o)*ni2+i];sum+=double(units_used?float(_Float16(v)):v)*in[r*ni2+i];}
                ref[ids[r]*stride+o]=sum;
            }
            bad+=compare(got,ref,units_used?"compact empty/nonzero/stride, matrix units":"compact empty/nonzero/stride",1e-4f);
            if(!units_used){if(!optimized)old=got;else bad+=compare(got,old,"compact against plain",0);}
        }
    }
    // 70 rows for each of two matrices: 64-row tiles, the second one partly empty, and ids that scatter every row
    for(int type:{12,18,23}) {
        const int ni3=512,no3=100,rows3=140;
        auto w=values(2*no3*ni3,.5f),x3=values(rows3*ni3,.7f);std::vector<float> imp(ni3,1);
        auto ty=ggml_type(type);ggml_quantize_init(ty);
        size_t wb=k::native_mmvq_weight_bytes(type,ni3,no3);
        std::vector<unsigned char> packed(wb*2);ggml_quantize_chunk(ty,w.data(),packed.data(),0,2*no3,ni3,imp.data());
        std::vector<float> dwf(w.size());ggml_get_type_traits(ty)->to_float(packed.data(),dwf.data(),dwf.size());
        std::vector<int32_t> bounds={0,70,140},ids(rows3);for(int r=0;r<rows3;++r)ids[r]=(r*37)%rows3;
        Buffer dw(packed.size()),dx3(x3),qx(p::Mmq::q8_bytes(rows3,ni3)),out(rows3*no3*4),bd(12),id(rows3*4);
        dw.put(packed.data(),packed.size());bd.put(bounds.data(),12);id.put(ids.data(),rows3*4);
        for(bool units:{false,true}) {
            g::metal::set_optimize(true);p::metal_detail::allow_mm_rows(units);
            p::Mmq::quantize(dx3.as(),nullptr,qx.ptr,type,ni3,ni3,rows3,stream);
            p::Mmq mmq;mmq.mul(dw.ptr,type,no3,ni3,wb,2,qx.ptr,bd.as<int32_t>(),id.as<int32_t>(),rows3,70,out.as(),no3,stream);
            ck(g::stream_synchronize(stream));auto got=out.read(rows3*no3);
            const bool units_used=p::metal_detail::mm_rows(ni3);
            auto in=mmq_rows(qx,x3,ni3,rows3);std::vector<float> ref(rows3*no3);
            for(int r=0;r<rows3;++r)for(int o=0;o<no3;++o){
                const int e=r<70?0:1;double sum=0;
                for(int i=0;i<ni3;++i){float v=dwf[(e*no3+o)*ni3+i];sum+=double(units_used?float(_Float16(v)):v)*in[r*ni3+i];}
                ref[ids[r]*no3+o]=sum;
            }
            bad+=compare(got,ref,units_used?"70 rows a matrix, matrix units":"70 rows a matrix",1e-4f);
        }
    }
    // A dense call of 200 rows: on the matrix units the matrix goes to fp16 once (poly_dq) and poly_gemm multiplies;
    // 100 outputs, so the last tile of them is partly past the matrix
    for(int type:{2,3,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,29,39,42}) {
        const int ni4=512,no4=100,rows4=200;
        auto w=values(no4*ni4,.5f),x4=values(rows4*ni4,.7f);std::vector<float> imp(ni4,1);
        auto ty=ggml_type(type);ggml_quantize_init(ty);
        size_t wb=k::native_mmvq_weight_bytes(type,ni4,no4);
        std::vector<unsigned char> packed(wb);ggml_quantize_chunk(ty,w.data(),packed.data(),0,no4,ni4,imp.data());
        std::vector<float> dwf(w.size());ggml_get_type_traits(ty)->to_float(packed.data(),dwf.data(),dwf.size());
        Buffer dw(packed.size()),dx4(x4),qx(p::Mmq::q8_bytes(rows4,ni4)),out(rows4*no4*4);
        dw.put(packed.data(),packed.size());
        g::metal::set_optimize(true);p::metal_detail::allow_mm_rows(true);
        p::Mmq::quantize(dx4.as(),nullptr,qx.ptr,type,ni4,ni4,rows4,stream);
        p::Mmq mmq;mmq.mul(dw.ptr,type,no4,ni4,0,1,qx.ptr,nullptr,nullptr,rows4,rows4,out.as(),no4,stream);
        ck(g::stream_synchronize(stream));
        const bool units_used=p::metal_detail::mm_rows(ni4);
        auto in=mmq_rows(qx,x4,ni4,rows4);std::vector<float> ref(rows4*no4);
        for(int r=0;r<rows4;++r)for(int o=0;o<no4;++o){
            double sum=0;
            for(int i=0;i<ni4;++i){float v=dwf[o*ni4+i];sum+=double(units_used?float(_Float16(v)):v)*in[r*ni4+i];}
            ref[r*no4+o]=sum;
        }
        char label[64];std::snprintf(label,sizeof label,"%s dense 200 rows%s",ggml_type_name(ty),units_used?", matrix units":"");
        bad+=compare(out.read(rows4*no4),ref,label,1e-4f);
    }
    // gemm_f32 (a router's fp32 matrix) for a batch: 64 and 100 outputs
    for(int no5:{64,100}) {
        const int ni5=512,rows5=200;
        auto w=values(no5*ni5,.3f),x5=values(rows5*ni5,.9f);
        Buffer dw(w),dx5(x5),out(rows5*no5*4);
        g::metal::set_optimize(true);p::metal_detail::allow_mm_rows(true);
        void* blas=p::blas_create(stream);
        p::gemm_f32(blas,dw.as(),dx5.as(),out.as(),ni5,no5,no5,rows5);ck(g::stream_synchronize(stream));p::blas_destroy(blas);
        std::vector<float> ref(rows5*no5);
        for(int r=0;r<rows5;++r)for(int o=0;o<no5;++o){double sum=0;for(int i=0;i<ni5;++i)sum+=double(w[o*ni5+i])*x5[r*ni5+i];ref[r*no5+o]=sum;}
        char label[64];std::snprintf(label,sizeof label,"gemm_f32 %d outputs",no5);
        bad+=compare(out.read(rows5*no5),ref,label,1e-5f);
    }
    p::metal_detail::allow_mm_rows(true);
    ck(g::stream_synchronize(stream));ck(g::stream_destroy(stream));ggml_quantize_free();return bad?1:0;
}
