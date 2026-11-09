#include <metal_stdlib>
using namespace metal;

// Mirrors BasicArgs in basic.cpp; device pointers are Metal gpuAddress values.
struct BasicArgs {
    device float* p[4];
    ulong count;
    int n, rows, stride, kind;
    float scale, eps;
};

inline float block_sum(float v, threadgroup float* partial, uint tid, uint nt) {
    for (uint k = 16; k; k >>= 1) v += simd_shuffle_xor(v, k);
    if ((tid & 31) == 0) partial[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint i = 0; i < nt / 32; ++i) sum += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sum;
}

kernel void poly_norm(constant BasicArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    device const float* x = a.p[0] + ulong(row) * a.n;
    device float* y = a.p[2] + ulong(row) * (a.stride ? a.stride : a.n);
    float mean = 0;
    if (a.kind) {
        float s = 0;
        for (int i = tid; i < a.n; i += nt) s += x[i];
        mean = block_sum(s, partial, tid, nt) / a.n;
    }
    float ss = 0;
    for (int i = tid; i < a.n; i += nt) { float z = x[i] - mean; ss += z * z; }
    float inv = rsqrt(block_sum(ss, partial, tid, nt) / a.n + a.eps);
    for (int i = tid; i < a.n; i += nt)
        y[i] = (x[i] - mean) * inv * (a.p[1] ? a.p[1][i] : 1.0f) + (a.p[3] ? a.p[3][i] : 0.0f);
}

kernel void poly_gemv(constant BasicArgs& a [[buffer(0)]], uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float partial[32];
    const ulong o = group.x, token = group.y;
    device const float* x = a.p[1] + (token * (a.rows / a.stride) + o / a.stride) * a.n;
    float acc = 0;
    for (int i = tid; i < a.n; i += nt) {
        ulong j = o * a.n + i;
        float w = a.kind ? as_type<float>(uint(reinterpret_cast<device ushort*>(a.p[0])[j]) << 16) : a.p[0][j];
        acc += w * x[i];
    }
    float sum = block_sum(acc, partial, tid, nt);
    if (tid == 0) a.p[2][token * a.rows + o] = sum;
}

kernel void poly_elementwise(constant BasicArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    if (a.kind == 0) a.p[1][i] = as_type<float>(uint(reinterpret_cast<device ushort*>(a.p[0])[i]) << 16);
    else if (a.kind == 1) a.p[1][i] = a.p[0][i] * a.scale;
    else if (a.kind == 2) a.p[1][i] += a.p[0][i] * a.scale;
    else if (a.kind == 3) a.p[0][i] = a.scale * tanh(a.p[0][i] / a.scale);
    else if (a.kind == 4) a.p[1][i] += a.p[0][i % a.n];
    else if (a.kind == 5 || a.kind == 6) {
        ulong at = a.p[1] ? i : (i / a.n) * 2 * a.n + i % a.n;
        float g = a.p[0][at], u = a.p[1] ? a.p[1][i] : a.p[0][at+a.n];
        if (a.kind == 5) { g=min(g,a.scale); u=clamp(u,-a.scale,a.scale); a.p[2][i]=g/(1.0f+exp(-g))*u; }
        else a.p[2][i]=0.5f*g*(1.0f+tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)))*u;
    } else if (a.kind == 7) {
        ulong row=i/a.n, col=i%a.n; float sum=0;
        for (int k=0;k<a.stride;++k) sum += a.p[1][row*a.stride+k]*a.p[0][(row*a.stride+k)*a.n+col];
        a.p[2][i]+=sum;
    } else if (a.kind == 8) a.p[0][i] /= 1.0f+exp(-a.p[1][(i/a.n)*a.stride+(a.rows?0:i%a.n)]);
    else if (a.kind == 9) {
        int pair = i < uint(a.n) ? 0 : 2; ulong j = i < uint(a.n) ? i : i-a.n;
        reinterpret_cast<device uint*>(a.p[pair+1])[j]=reinterpret_cast<device uint*>(a.p[pair])[j];
    }

}

struct ConvArgs {
    device const float* x;
    device float* state;
    device const float* w;
    device float* out;
    device float* snap;
    ulong stride;
    int channels, rows, kind;
    uint snaps;
};
// A channel's input at row r: B * x (kind 0) or x (kind 1); before the call's first row, the state's (it holds the
// `past` values before row 0, oldest first)
inline float conv_in(constant ConvArgs& a, uint c, int r, int past) {
    if (r < 0) return a.state[ulong(c) * past + past + r];
    device const float* x = a.x + ulong(r) * a.channels * (a.kind ? 1 : 3);
    return a.kind ? x[c] : x[c] * x[2 * a.channels + c];
}
// A thread to each channel of each row: a row reads the inputs of the `past` rows before it, so the rows need not
// go in order.  poly_conv_state then moves the state on (here a row's thread could still read the old one).
kernel void poly_conv(constant ConvArgs& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint c = tg.x * shape.x + tid;
    const int row = int(tg.y);
    if (c >= uint(a.channels)) return;
    const int past = a.kind ? 3 : 2;
    device const float* w = a.w + ulong(c) * (past + 1);
    const float s0 = conv_in(a, c, row - past, past), s1 = conv_in(a, c, row - past + 1, past);
    const float s2 = a.kind ? conv_in(a, c, row - 1, past) : 0, v = conv_in(a, c, row, past);
    float y;
    if (a.kind) {
        y = s0*w[0] + s1*w[1] + s2*w[2] + v*w[3];
        y = y / (1.0f + exp(-y));
    } else {
        y = (s0*w[0] + s1*w[1] + v*w[2]) * a.x[ulong(row) * a.channels * 3 + a.channels + c];
    }
    a.out[ulong(row)*a.channels+c]=y;
    if (a.snap && row < 32 && ((a.snaps >> row) & 1u)) {   // the state after this row
        device float* dst = a.snap + ulong(popcount(a.snaps & ((1u << row)-1u))) * a.stride + ulong(c)*past;
        if (a.kind) { dst[0]=s1; dst[1]=s2; dst[2]=v; }
        else { dst[0]=s1; dst[1]=v; }
    }
}
// The state after the call's last row: the last `past` inputs, the old state's where the call has fewer rows (read
// before they are written over: state[i] takes old state[rows + i])
kernel void poly_conv_state(constant ConvArgs& a [[buffer(0)]], uint c [[thread_position_in_grid]]) {
    if (c >= uint(a.channels)) return;
    const int past = a.kind ? 3 : 2;
    for (int i = 0; i < past; ++i) a.state[ulong(c) * past + i] = conv_in(a, c, a.rows - past + i, past);
}

struct RouteArgs {
    device const float* logits;
    device const float* bias;
    device const float* escale;
    device int* ids;
    device float* weights;
    int experts, used, rows, gate;
    float in_scale, w_scale;
    int norm;
};
kernel void poly_route(constant RouteArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    float p[16], pick[16], mx=-INFINITY;
    const int per = a.experts <= 64 ? 2 : a.experts <= 128 ? 4 : a.experts <= 256 ? 8 : 16;
    for(int k=0;k<per;++k) {
        int e=lane+32*k;
        p[k]=e<a.experts ? a.logits[ulong(row)*a.experts+e]*a.in_scale : -INFINITY;
        mx=max(mx,p[k]);
    }
    if(a.gate==0) {
        for(int o=16;o;o>>=1) mx=max(mx,simd_shuffle_xor(mx,o));
        float sum=0;
        for(int k=0;k<per;++k) { p[k]=int(lane)+32*k<a.experts ? exp(p[k]-mx):0; sum+=p[k]; }
        for(int o=16;o;o>>=1) sum+=simd_shuffle_xor(sum,o);
        const float inv=1.0f/sum;
        for(int k=0;k<per;++k) p[k]*=inv;
    } else if(a.gate==3) {
        // sigmoid_logit_add (Kolibri 1): the weights are the sigmoid of the scores, but the pick is on the scores and
        // the bias, so it is taken before the scores turn to weights
        for(int k=0;k<per;++k) {
            int e=lane+32*k;
            pick[k]=e<a.experts ? p[k]+(a.bias?a.bias[e]:0): -INFINITY;
            p[k]=e<a.experts ? 1.0f/(1.0f+exp(-p[k])) : -INFINITY;
        }
    } else {
        for(int k=0;k<per;++k) { float x=p[k]; p[k]=a.gate==1 ? 1.0f/(1.0f+exp(-x)) : sqrt(x>20 ? x:log(1.0f+exp(x))); }
    }
    if(a.gate!=3) for(int k=0;k<per;++k) {
        int e=lane+32*k;
        pick[k]=e<a.experts ? p[k]+(a.bias?a.bias[e]:0): -INFINITY;
    }
    float sum=0, mine=0; int my_id=0;
    for(int j=0;j<a.used;++j) {
        float best=-INFINITY; int at=a.experts;
        for(int k=0;k<per;++k) if(pick[k]>best) {best=pick[k];at=lane+32*k;}
        for(int o=16;o;o>>=1) {
            float b=simd_shuffle_xor(best,o); int id=simd_shuffle_xor(at,o);
            if(b>best || (b==best && id<at)) {best=b;at=id;}
        }
        float pw=0;
        for(int k=0;k<per;++k) if(int(lane)+32*k==at) {pw=p[k];pick[k]=-INFINITY;}
        pw=simd_shuffle(pw,at%32); sum+=pw;
        if(int(lane)==j) {mine=pw;my_id=at;}
    }
    if(int(lane)>=a.used) return;
    if(a.norm) mine/=max(sum,6.103515625e-5f);
    if(a.escale) mine*=a.escale[my_id];
    if(a.w_scale!=0) mine*=a.w_scale;
    a.ids[ulong(row)*a.used+lane]=my_id;
    a.weights[ulong(row)*a.used+lane]=mine;
}

struct ScaledArgs { device const float* x[32]; device float* acc[32]; float w[32]; int count,n; };
kernel void poly_scaled(constant ScaledArgs& a [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(a.n)) return;
    for(int k=0;k<a.count;++k) a.acc[k][i]+=a.x[k][i]*a.w[k];
}
// The likeliest token and its probability: one threadgroup (up to 1,024 threads), its SIMD groups' results joined
// through threadgroup memory; a vocabulary of 248K was 1.8 ms on 32 threads
kernel void poly_likeliest(constant BasicArgs& a [[buffer(0)]],uint tid [[thread_index_in_threadgroup]],
                           uint nt [[threads_per_threadgroup]],uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tm[32],ts[32];
    threadgroup int ta[32];
    const uint groups=(nt+31)/32;
    float m=-INFINITY;
    for(int i=tid;i<a.n;i+=nt) m=max(m,a.p[0][i]);
    m=simd_max(m);
    if(!lane) tm[sg]=m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    m=tm[0];
    for(uint g=1;g<groups;++g) m=max(m,tm[g]);
    float sum=0; int at=a.n;
    for(int i=tid;i<a.n;i+=nt) { float v=a.p[0][i]; if(v==m) at=min(at,i); if(v>m-20) sum+=exp(v-m); }
    sum=simd_sum(sum); at=simd_min(at);
    if(!lane) { ts[sg]=sum; ta[sg]=at; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid) return;
    sum=0; at=a.n;
    for(uint g=0;g<groups;++g) { sum+=ts[g]; at=min(at,ta[g]); }
    reinterpret_cast<device int*>(a.p[1])[0]=at<a.n?at:0; a.p[1][1]=1.0f/sum;
}
