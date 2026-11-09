#include <metal_stdlib>
using namespace metal;
struct Args {
    device void* p[10]; long pos;
    int dim,heads,kv,rows,window,ring,whole,stride,rot,sections,kind;
    float scale,eps;
};
kernel void poly_store(constant Args& a [[buffer(0)]],uint i [[thread_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    if(i>=uint(a.dim)*a.rows) return;
    long pos=(a.p[3]?*static_cast<device long*>(a.p[3]):a.pos)+i/a.dim;
    if(a.ring) pos%=a.ring;
    ulong j=ulong(pos)*a.dim+i%a.dim;
    float x=static_cast<device float*>(a.p[0])[i];
    if(!a.kind) static_cast<device half*>(a.p[1])[j]=half(x);
    else {
        float d=simd_max(abs(x))/127.0f;
        static_cast<device char*>(a.p[1])[j]=char(d?round(x/d):0);
        if(!lane) static_cast<device half*>(a.p[2])[j/32]=half(d);
    }
}
kernel void poly_rope(constant Args& a [[buffer(0)]],uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    device const float* x=static_cast<device float*>(a.p[0])+ulong(head)*a.stride;
    device const float* w=static_cast<device float*>(a.p[1]);
    device float* y=static_cast<device float*>(a.p[2])+ulong(head)*a.dim;
    float values[32]; float sum=0;
    for(int i=lane;i<a.dim;i+=32) { float v=x[i];sum+=v*v; }
    float inv=rsqrt(simd_sum(sum)/a.dim+a.eps);
    int token=head/a.heads,half_rot=a.rot/2;
    long own=(a.p[5]?*static_cast<device long*>(a.p[5]):a.pos)+token;
    for(int i=lane;i<a.dim;i+=32) {
        float value=x[i]*inv*(w?w[i]:1.0f);
        if(i<a.rot) {
            int j=i%half_rot;
            float position=a.p[4]?float(static_cast<device int*>(a.p[4])[token*3+(j%a.sections)%3]):float(own);
            float theta=position*pow(a.scale,float(j));
            if(a.p[3]) theta/=static_cast<device float*>(a.p[3])[j];
            float x0=x[j]*inv*(w?w[j]:1.0f),x1=x[j+half_rot]*inv*(w?w[j+half_rot]:1.0f);
            value=i<half_rot?x0*cos(theta)-x1*sin(theta):x0*sin(theta)+x1*cos(theta);
        }
        values[i/32]=value;
    }
    threadgroup_barrier(mem_flags::mem_device); // x and y may alias
    for(int i=lane;i<a.dim;i+=32) y[i]=values[i/32];
}
inline float cache_value(device void* raw,device void* codes,device void* scales,ulong at) {
    return codes?float(static_cast<device char*>(codes)[at])*float(static_cast<device half*>(scales)[at/32]):float(static_cast<device half*>(raw)[at]);
}
// One threadgroup per query head and token, with an online softmax, so the scratch does not grow with the context.
// Its SIMD groups take every g-th key (g = the SIMD groups in it), each with a softmax of its own, and the first
// merges the g of them through threadgroup memory: a long context's keys are read g at a time instead of one by
// one.  With one SIMD group the keys go in order, as they did before there were more.  PER: a lane's share of a
// head (head dimension / 32 at most), so a head of up to 256 values keeps 8 floats a lane and not 32.
// With `stride` splits (p[9] their scratch), threadgroup z takes the z-th of that many runs of the token's keys
// and leaves its maximum, denominator and unnormalized sums there for poly_attention_join: a SIMD group walks its
// keys at about 3 us a key at heads of 256 (M5 Ultra), so a token's keys go to many threadgroups.
template<int PER>
kernel void poly_attention(constant Args& a [[buffer(0)]],threadgroup float* part [[threadgroup(0)]],
                           uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],uint groups [[simdgroups_per_threadgroup]]) {
    int h=group.x,t=group.y,kh=h/(a.heads/a.kv);
    device const float* q=static_cast<device float*>(a.p[0])+(ulong(t)*a.heads+h)*a.dim;
    device float* out=static_cast<device float*>(a.p[3])+(ulong(t)*a.heads+h)*a.dim;
    long start=a.p[4]?*static_cast<device long*>(a.p[4]):a.pos,own=start+t;
    long lo=a.window>0?max(0L,own-a.window+1):0,hi=a.whole?start+a.rows-1:own;
    const int splits=max(a.stride,1);
    device float* run=splits>1?static_cast<device float*>(a.p[9])+((ulong(t)*a.heads+h)*splits+group.z)*(a.dim+2):nullptr;
    if(run) {   // this threadgroup's run of the keys (none: an empty run)
        const long span=(hi-lo+splits)/splits,first=lo+long(group.z)*span;
        hi=min(hi,first+span-1);lo=first;
    }
    // the SIMD groups that take keys: one to each 128 of this token's, so that a short context reads its keys in
    // order in one group and skips the merge, as before (the same for the whole threadgroup: one token); a run of
    // a split, one to each 8
    const long per=run?8:128;
    const uint used=uint(clamp((hi-lo+per)/per,1L,long(groups)));
    if(used==1 && sg) return;
    float acc[PER]={0},query[PER];
    for(int d=lane;d<a.dim;d+=32) query[d/32]=q[d];
    float maximum=-INFINITY,denom=0;
    for(long p=lo+long(sg);sg<used && p<=hi;p+=long(used)) {
        ulong offset=(ulong(a.ring?p%a.ring:p)*a.kv+kh)*a.dim;
        float dot=0;
        for(int d=lane;d<a.dim;d+=32) dot+=query[d/32]*cache_value(a.p[1],a.p[5],a.p[6],offset+d);
        dot=simd_sum(dot)*a.scale;
        float next=max(maximum,dot),keep=exp(maximum-next),weight=exp(dot-next);
        denom=denom*keep+weight; maximum=next;
        for(int d=lane;d<a.dim;d+=32) acc[d/32]=acc[d/32]*keep+weight*cache_value(a.p[2],a.p[7],a.p[8],offset+d);
    }
    if(used==1) {
        if(run) {
            if(!lane) { run[0]=maximum;run[1]=denom; }
            for(int d=lane;d<a.dim;d+=32) run[2+d]=acc[d/32];
            return;
        }
        for(int d=lane;d<a.dim;d+=32) out[d]=acc[d/32]/denom;
        return;
    }
    // part: each SIMD group's maximum, its denominator, then its dim sums; a group past `used` writes nothing
    threadgroup float* m=part;
    threadgroup float* l=part+groups;
    threadgroup float* v=part+2*groups;
    if(sg<used) {
        if(lane==0) {m[sg]=maximum;l[sg]=denom;}
        for(int d=lane;d<a.dim;d+=32) v[sg*a.dim+d]=acc[d/32];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(sg) return;
    float top=-INFINITY,total=0;
    for(uint g=0;g<used;++g) top=max(top,m[g]);
    for(uint g=0;g<used;++g) total+=l[g]*exp(m[g]-top);
    if(run && !lane) { run[0]=top;run[1]=total; }
    for(int d=lane;d<a.dim;d+=32) {
        float sum=0;
        for(uint g=0;g<used;++g) sum+=v[g*a.dim+d]*exp(m[g]-top);
        if(run) run[2+d]=sum;
        else out[d]=sum/total;
    }
}
// The splits' runs of a head and token (poly_attention with stride splits) into its output; an empty run has a
// maximum of -infinity and adds nothing.  grid (heads, rows), 32 threads.
kernel void poly_attention_join(constant Args& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const int h=group.x,t=group.y,splits=a.stride,n=a.dim+2;
    device const float* run=static_cast<device float*>(a.p[9])+(ulong(t)*a.heads+h)*splits*n;
    device float* out=static_cast<device float*>(a.p[3])+(ulong(t)*a.heads+h)*a.dim;
    float top=-INFINITY,total=0;
    for(int s=0;s<splits;++s) top=max(top,run[s*n]);
    for(int s=0;s<splits;++s) if(run[s*n+1]>0) total+=run[s*n+1]*exp(run[s*n]-top);
    for(int d=lane;d<a.dim;d+=32) {
        float sum=0;
        for(int s=0;s<splits;++s) if(run[s*n+1]>0) sum+=run[s*n+2+d]*exp(run[s*n]-top);
        out[d]=sum/total;
    }
}
typedef decltype(poly_attention<32>) poly_attention_t;
template [[host_name("poly_attention")]] kernel poly_attention_t poly_attention<32>;
template [[host_name("poly_attention_256")]] kernel poly_attention_t poly_attention<8>;
