#include <metal_stdlib>
using namespace metal;
struct Args { device float* p[10]; ulong stride; int dim,heads,keys,rows; float eps; uint snaps; };
inline float warp_sum(float v) { for(uint k=16;k;k>>=1) v+=simd_shuffle_xor(v,k);return v; }
kernel void poly_gdn_norm(constant Args& a [[buffer(0)]],uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    ulong row_n=ulong(2*a.keys+a.heads)*a.dim;
    device float* q=a.p[0]+(head/a.keys)*row_n+(head%a.keys)*a.dim;
    device float* k=q+a.keys*a.dim;
    float qs=0,ks=0;
    for(int i=lane;i<a.dim;i+=32) { qs+=q[i]*q[i];ks+=k[i]*k[i]; }
    float qi=rsqrt(warp_sum(qs)+a.eps)*rsqrt(float(a.dim)),ki=rsqrt(warp_sum(ks)+a.eps);
    for(int i=lane;i<a.dim;i+=32) { q[i]*=qi;k[i]*=ki; }
}
kernel void poly_gdn_gates(constant Args& a [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(a.rows*a.heads)) return;
    float g=a.p[1][i]+a.p[3][i%a.heads];
    a.p[1][i]=exp((g>20?g:log(1.0f+exp(g)))*a.p[4][i%a.heads]);
    a.p[2][i]=1.0f/(1.0f+exp(-a.p[2][i]));
}
kernel void poly_gdn_state(constant Args& a [[buffer(0)]],uint col [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    int h=col/a.dim,j=col%a.dim,hk=h%a.keys;
    ulong keys=ulong(a.keys)*a.dim,inner=ulong(a.heads)*a.dim,mine=ulong(col)*a.dim;
    float state[8];
    for(int i=lane;i<a.dim;i+=32) state[i/32]=a.p[5][mine+i];
    for(int row=0;row<a.rows;++row) {
        device const float* x=a.p[0]+ulong(row)*(2*keys+inner);
        device const float* q=x+hk*a.dim; device const float* k=q+keys;
        float decay=a.p[1][row*a.heads+h],sk=0;
        for(int i=lane;i<a.dim;i+=32) sk+=state[i/32]*decay*k[i];
        float delta=(x[2*keys+h*a.dim+j]-warp_sum(sk))*a.p[2][row*a.heads+h],o=0;
        bool save=a.p[9] && row<32 && ((a.snaps>>row)&1u);
        ulong snap_at=save?ulong(popcount(a.snaps&((1u<<row)-1)))*a.stride+mine:0;
        for(int i=lane;i<a.dim;i+=32) {
            float v=state[i/32]*decay+k[i]*delta; state[i/32]=v;o+=v*q[i];
            if(save) a.p[9][snap_at+i]=v;
        }
        o=warp_sum(o); if(!lane) a.p[8][ulong(row)*inner+h*a.dim+j]=o;
    }
    for(int i=lane;i<a.dim;i+=32) a.p[5][mine+i]=state[i/32];
}
// poly_gdn_state for a prompt batch, heads of D values (after metal/kernels/gdn_prompt.metal's gdn_scan_r): a lane to
// each column of a head's state and IB SIMD groups to its rows, D / IB of them in each lane's registers.  A row's
// three dots (S k, S q of the old state, and k q) are summed over the groups through threadgroup memory, double
// buffered so that a row takes one barrier; the output is decay (S q) + delta (k q), which is S_new q.  The rows' q, k
// (the head's key head's), v, decay and beta are staged TB rows at a time instead of read again by every column.
// grid: heads * D / 32 threadgroups of IB * 32 (a threadgroup's 32 columns lie in one head).
template<int D,int IB,int TB>
kernel void poly_gdn_scan(constant Args& a [[buffer(0)]],uint tg [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]]) {
    constexpr int R=D/IB,R4=R/4;
    threadgroup float4 qs[TB][D/4],ks[TB][D/4];
    threadgroup float4 red[2][IB][32];
    threadgroup float vs[TB][32];
    threadgroup float2 gb[TB];
    const uint c0=tg*32,col=c0+lane,h=c0/D;
    const int hk=int(h)%a.keys,i0=int(sg)*R;
    const ulong keys=ulong(a.keys)*D,inner=ulong(a.heads)*D,row_n=2*keys+inner,mine=ulong(col)*D+i0;
    float st[R];
#pragma clang loop unroll(full)
    for(int i=0;i<R;++i) st[i]=a.p[5][mine+i];
    for(int r0=0;r0<a.rows;r0+=TB) {
        const int nb=min(TB,a.rows-r0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int idx=tid;idx<nb*(D/4);idx+=IB*32) {
            const int t=idx/(D/4),w=idx%(D/4);
            device const float4* x=reinterpret_cast<device const float4*>(a.p[0]+ulong(r0+t)*row_n+ulong(hk)*D);
            qs[t][w]=x[w];ks[t][w]=x[keys/4+w];
        }
        for(int idx=tid;idx<nb*32;idx+=IB*32) {
            const int t=idx/32,c=idx%32,row=r0+t;
            vs[t][c]=a.p[0][ulong(row)*row_n+2*keys+c0+c];
            if(!c) gb[t]=float2(a.p[1][row*a.heads+h],a.p[2][row*a.heads+h]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for(int t=0;t<nb;++t) {
            const int row=r0+t,slot=row&1;
            float ka=0,qa=0,kq=0;
#pragma clang loop unroll(full)
            for(int w=0;w<R4;++w) {
                const float4 k4=ks[t][i0/4+w],q4=qs[t][i0/4+w];
                ka=fma(st[4*w],k4.x,ka);ka=fma(st[4*w+1],k4.y,ka);ka=fma(st[4*w+2],k4.z,ka);ka=fma(st[4*w+3],k4.w,ka);
                qa=fma(st[4*w],q4.x,qa);qa=fma(st[4*w+1],q4.y,qa);qa=fma(st[4*w+2],q4.z,qa);qa=fma(st[4*w+3],q4.w,qa);
                kq=fma(k4.x,q4.x,kq);kq=fma(k4.y,q4.y,kq);kq=fma(k4.z,q4.z,kq);kq=fma(k4.w,q4.w,kq);
            }
            red[slot][sg][lane]=float4(ka,qa,kq,0);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float4 s=red[slot][0][lane];
#pragma clang loop unroll(full)
            for(int b=1;b<IB;++b) s+=red[slot][b][lane];
            const float decay=gb[t].x,delta=(vs[t][lane]-decay*s.x)*gb[t].y;
            if(!sg) a.p[8][ulong(row)*inner+col]=fma(decay,s.y,delta*s.z);
#pragma clang loop unroll(full)
            for(int w=0;w<R4;++w) {
                const float4 k4=ks[t][i0/4+w];
                st[4*w]=fma(decay,st[4*w],k4.x*delta);st[4*w+1]=fma(decay,st[4*w+1],k4.y*delta);
                st[4*w+2]=fma(decay,st[4*w+2],k4.z*delta);st[4*w+3]=fma(decay,st[4*w+3],k4.w*delta);
            }
            if(a.p[9] && row<32 && ((a.snaps>>row)&1u)) {
                device float* snap=a.p[9]+ulong(popcount(a.snaps&((1u<<row)-1)))*a.stride+mine;
#pragma clang loop unroll(full)
                for(int i=0;i<R;++i) snap[i]=st[i];
            }
        }
    }
#pragma clang loop unroll(full)
    for(int i=0;i<R;++i) a.p[5][mine+i]=st[i];
}
template [[host_name("poly_gdn_scan_128")]] kernel decltype(poly_gdn_scan<128,4,16>) poly_gdn_scan<128,4,16>;
template [[host_name("poly_gdn_scan_64")]] kernel decltype(poly_gdn_scan<64,2,16>) poly_gdn_scan<64,2,16>;
kernel void poly_gdn_out(constant Args& a [[buffer(0)]],uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    ulong base=ulong(head)*a.dim;float ss=0;
    for(int i=lane;i<a.dim;i+=32) {float v=a.p[8][base+i];ss+=v*v;}
    float inv=rsqrt(warp_sum(ss)/a.dim+a.eps);
    for(int i=lane;i<a.dim;i+=32) {float z=a.p[6][base+i];a.p[8][base+i]*=inv*a.p[7][i]*(z/(1.0f+exp(-z)));}
}
