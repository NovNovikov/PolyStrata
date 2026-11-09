#include "test_utils.hpp"
#include "strata/poly/kernels.hpp"
#include <algorithm>
namespace p=strata::poly;
int main(){
    int bad=0; g::Stream s;ck(g::stream_create(&s));
    // One attend() call against a CPU reference over the cache as store wrote it: `total` rows, the last `rows` of
    // them the call's tokens.  recorded: captured, with the position on the card and the launch sized for `room` rows.
    // batch: attend_batch() instead, whose products take the queries and the weights in fp16 (flash.metal): the
    // reference takes the queries rounded so, and the weights' rounding is within the looser bound.
    auto attention=[&](int dim,bool quant,int window,int ring,bool whole,int heads,int kv,int rows,int total,
                       bool recorded,int room,bool batch=false){
        const int pos=total-rows,n=kv*dim;
        auto keys=values(size_t(total)*n),vals=values(size_t(total)*n,.63f),query=values(size_t(rows)*heads*dim,.42f);
        Buffer dk(keys),dv(vals),dq(query),kcache(size_t(total)*n*2),vcache(size_t(total)*n*2),kscale(size_t(total)*n/32*2),
               vscale(size_t(total)*n/32*2),out(size_t(rows)*heads*dim*4),at(8);
        long long position=0;at.put(&position,8);
        if(quant){p::store_q8_at(dk.as(),kcache.as<int8_t>(),kscale.as<uint16_t>(),n,at.as<long long>(),total,0,s);p::store_q8_at(dv.as(),vcache.as<int8_t>(),vscale.as<uint16_t>(),n,at.as<long long>(),total,0,s);}
        else {p::store_f16(dk.as(),kcache.as<uint16_t>(),n,0,total,s);p::store_f16_at(dv.as(),vcache.as<uint16_t>(),n,at.as<long long>(),total,0,s);}
        ck(g::stream_synchronize(s));
        // Independent CPU decoding of what store wrote, checked against FP16/Q8 rounding.
        auto decode=[&](Buffer& c,Buffer& scale,const std::vector<float>& original){
            auto half=c.read<_Float16>(size_t(total)*n);auto codes=c.read<int8_t>(size_t(total)*n);auto scales=scale.read<_Float16>(size_t(total)*n/32);
            std::vector<float> result(size_t(total)*n);
            for(int i=0;i<total*n;++i){
                if(!quant){bad+=float(half[i])!=float(_Float16(original[i]));result[i]=float(half[i]);}
                else {float mx=0;for(int j=i/32*32;j<i/32*32+32;++j)mx=std::max(mx,std::abs(original[j]));float d=mx/127;
                    bad+=codes[i]!=int(std::round(original[i]/d));bad+=float(scales[i/32])!=float(_Float16(d));result[i]=codes[i]*float(scales[i/32]);}
            }return result;
        };
        auto k=decode(kcache,kscale,keys),v=decode(vcache,vscale,vals);
        auto qr=query;if(batch)for(auto& x:qr)x=float(_Float16(x));
        if(batch&&quant){for(auto& x:k)x=float(_Float16(x));for(auto& x:v)x=float(_Float16(x));}
        // Ring reads are tested against explicitly wrapped physical cache indices.
        std::vector<float> ref(size_t(rows)*heads*dim);
        for(int t=0;t<rows;++t)for(int h=0;h<heads;++h){
            int lo=window?std::max(0,pos+t-window+1):0,hi=whole?total-1:pos+t;
            std::vector<double> scores;double mx=-1e100;
            for(int j=lo;j<=hi;++j){int off=((ring?j%ring:j)*kv+h/(heads/kv))*dim;double dot=0;for(int d=0;d<dim;++d)dot+=double(qr[(t*heads+h)*dim+d])*k[off+d];scores.push_back(dot*.125);mx=std::max(mx,dot*.125);}
            double denom=0;for(auto& a:scores){a=std::exp(a-mx);denom+=a;}
            for(int d=0;d<dim;++d){double sum=0;for(int j=lo;j<=hi;++j)sum+=scores[j-lo]*v[((ring?j%ring:j)*kv+h/(heads/kv))*dim+d];ref[(t*heads+h)*dim+d]=sum/denom;}
        }
        p::Kvq8 kq{quant?kcache.as<int8_t>():nullptr,kscale.as<uint16_t>()},vq{quant?vcache.as<int8_t>():nullptr,vscale.as<uint16_t>()};
        if(batch){
            p::attend_batch(nullptr,dq.as(),kcache.as<uint16_t>(),vcache.as<uint16_t>(),pos,rows,nullptr,0,nullptr,out.as(),dim,heads,kv,.125f,s,window,ring,whole,kq,vq);
            ck(g::stream_synchronize(s));
        } else if(recorded){
            position=pos;at.put(&position,8);
            ck(g::begin_capture(s));p::attend(dq.as(),kcache.as<uint16_t>(),vcache.as<uint16_t>(),0,rows,nullptr,2,out.as(),dim,heads,kv,.125f,s,window,ring,whole,at.as<long long>(),room,0,kq,vq);
            g::Graph graph;ck(g::end_capture(s,&graph));g::GraphExec exec;ck(g::graph_instantiate(&exec,graph));ck(g::graph_launch(exec,s));ck(g::stream_synchronize(s));
            g::graph_exec_destroy(exec);g::graph_destroy(graph);
        } else {
            p::attend(dq.as(),kcache.as<uint16_t>(),vcache.as<uint16_t>(),pos,rows,nullptr,2,out.as(),dim,heads,kv,.125f,s,window,ring,whole,nullptr,0,0,kq,vq);
            ck(g::stream_synchronize(s));
        }
        char label[96];
        std::snprintf(label,sizeof label,"%s dim %d%s window %d%s, %d keys%s",batch?"attend_batch":"attention",dim,quant?" q8":"",window,whole?" whole":"",total,recorded?" recorded":"");
        bad+=compare(out.read(size_t(rows)*heads*dim),ref,label,batch?2e-3f:3e-5f);
    };
    for(int dim:{32,64,128})for(bool quant:{false,true})for(int window:{0,7})for(bool whole:{false,true})
        attention(dim,quant,window,window?11:0,whole,4,2,3,19,true,19);
    // Long contexts, whose keys a call splits among up to 16 SIMD groups (attend() picks them from the keys)
    for(int dim:{64,128,256})for(bool quant:{false,true})for(int window:{0,700})for(bool recorded:{false,true})
        attention(dim,quant,window,window?1024:0,false,8,2,3,2600,recorded,4096);
    // 300 keys: a recorded launch has 16 SIMD groups and gives keys to 3 of them, a direct one has 2
    for(int dim:{128,256})for(bool quant:{false,true})for(bool recorded:{false,true})
        attention(dim,quant,0,0,false,8,2,3,300,recorded,4096);
    // A prompt batch: 70 tokens (16 a threadgroup, the last 6 alone) after 2,530 keys, and 40 from the first row
    for(int dim:{64,128,256})for(bool quant:{false,true})for(int window:{0,700})for(bool whole:{false,true}){
        attention(dim,quant,window,window?1024:0,whole,8,2,70,2600,false,0,true);
        attention(dim,quant,window?7:0,window?11:0,whole,8,2,40,40,false,0,true);
    }
    // Per-head RMS, NeoX pairs, frequencies, image axes, and an in-place output.
    for(bool axes:{false,true}){
        int dim=64,heads=2,rows=3,rot=32;auto x=values(dim*heads*rows),w=values(dim,.2f);for(auto& v:w)v+=1;
        std::vector<float> freq(rot/2,1.3f),ref(x.size());std::vector<int32_t> axis={3,2,1,4,2,2,5,3,2};
        Buffer dx(x),dw(w),df(freq),da(axis.size()*4);da.put(axis.data(),axis.size()*4);
        for(int h=0;h<heads*rows;++h){double ss=0;for(int i=0;i<dim;++i)ss+=double(x[h*dim+i])*x[h*dim+i];double inv=1/std::sqrt(ss/dim+1e-5);
            for(int i=0;i<dim;++i){double value=x[h*dim+i]*inv*w[i];if(i<rot){int j=i%(rot/2);double theta=(axes?axis[(h/heads)*3+(j%6)%3]:7+h/heads)*std::pow(.91,j)/freq[j];double a=x[h*dim+j]*inv*w[j],b=x[h*dim+j+rot/2]*inv*w[j+rot/2];value=i<rot/2?a*std::cos(theta)-b*std::sin(theta):a*std::sin(theta)+b*std::cos(theta);}ref[h*dim+i]=value;}}
        p::head_norm_rope(dx.as(),dim,dw.as(),dx.as(),dim,heads,rot,.91f,7,rows,1e-5f,s,df.as(),axes?da.as<int32_t>():nullptr,6);ck(g::stream_synchronize(s));bad+=compare(dx.read(x.size()),ref,"rope");
    }
    ck(g::stream_destroy(s));return bad?1:0;
}
