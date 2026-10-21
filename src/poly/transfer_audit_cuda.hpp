// Timings are stream intervals, not additive wall-time. No new host synchronization.
#pragma once
#include "transfer_audit.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <unordered_map>
#include <stdexcept>

namespace strata::poly::audit {
inline const char* direction(cudaMemcpyKind k) {
    switch(k){case cudaMemcpyHostToDevice:return "H2D";case cudaMemcpyDeviceToHost:return "D2H";
        case cudaMemcpyDeviceToDevice:return "D2D";case cudaMemcpyHostToHost:return "H2H";default:return "copy-default";}
}
inline bool capturing(cudaStream_t stream) {
    cudaStreamCaptureStatus status=cudaStreamCaptureStatusNone;
    const auto e=cudaStreamIsCapturing(stream,&status);
    if(e!=cudaSuccess)throw std::runtime_error("transfer audit: cannot query stream capture");
    return status!=cudaStreamCaptureStatusNone;
}
struct Pending { cudaEvent_t first=nullptr,last=nullptr; Key k; int device=0; bool sent=false; };
struct GpuState {
    std::mutex mutex; std::vector<Pending> pending,free;
    std::vector<std::pair<int,cudaStream_t>> streams;
    struct GraphCopy { cudaMemcpyKind kind; size_t bytes; std::string label; };
    std::map<uintptr_t,std::vector<GraphCopy>> graphs;
    std::map<uintptr_t,std::vector<GraphCopy>> custom_templates;
};
inline GpuState& gpu_state(){static GpuState s;return s;}
inline thread_local std::vector<GpuState::GraphCopy> custom_capture;
inline void capture_begin(){if(enabled())custom_capture.clear();}
inline void capture_end(cudaGraph_t graph){if(enabled()){auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);s.custom_templates[(uintptr_t)graph]=custom_capture;custom_capture.clear();}}
inline void kernel_copy(const char* f,int l,const char* name,size_t bytes,cudaMemcpyKind kind,cudaStream_t stream) {
    if(!enabled())return;bool cap=capturing(stream);std::string op=std::string("kernel-")+direction(kind)+"-"+name;
    add(key(f,l,(op+(cap?"-capture-template":"")).c_str(),(uintptr_t)stream),bytes);
    if(cap)custom_capture.push_back({kind,bytes,op});
}
inline bool device_pointer(const void* ptr) {
    static thread_local std::unordered_map<const void*,bool> types;
    auto it=types.find(ptr);if(it!=types.end())return it->second;
    cudaPointerAttributes a{};if(cudaPointerGetAttributes(&a,ptr)!=cudaSuccess)throw std::runtime_error("transfer audit: copy2 pointer classification failed");
#ifdef HIP_VERSION
    bool device=a.memoryType==cudaMemoryTypeDevice;
#else
    bool device=a.type==cudaMemoryTypeDevice;
#endif
    types[ptr]=device;return device;
}
inline cudaMemcpyKind pointer_direction(const void* src,const void* dst){bool s=device_pointer(src),d=device_pointer(dst);return s?(d?cudaMemcpyDeviceToDevice:cudaMemcpyDeviceToHost):(d?cudaMemcpyHostToDevice:cudaMemcpyHostToHost);}
inline void watch(cudaStream_t stream) {
    if(!enabled()||!stream)return;int d=0;cudaGetDevice(&d);auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);
    auto item=std::make_pair(d,stream);if(std::find(s.streams.begin(),s.streams.end(),item)==s.streams.end())s.streams.push_back(item);
}
inline cudaError_t create_stream(cudaStream_t* stream,unsigned flags) { auto e=cudaStreamCreateWithFlags(stream,flags);if(e==cudaSuccess)watch(*stream);return e; }
inline int busy() {
    int d=0;cudaGetDevice(&d);auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);int n=0;
    for(auto item:s.streams)if(item.first==d){const auto e=cudaStreamQuery(item.second);if(e==cudaErrorNotReady)++n;else if(e!=cudaSuccess)count("stream-query-failed");}return n;
}
inline void collect() {
    if(!enabled())return;int d=0;cudaGetDevice(&d);auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);
    for(size_t i=0;i<s.pending.size();){auto& p=s.pending[i];if(p.device!=d||!p.sent||cudaEventQuery(p.last)!=cudaSuccess){++i;continue;}
        float ms=0;if(cudaEventElapsedTime(&ms,p.first,p.last)==cudaSuccess)gpu(p.k,ms);else count("gpu-time-failed");
        s.free.push_back(p);s.pending.erase(s.pending.begin()+i);
    }
}
struct GpuSpan {
    Pending p; cudaStream_t stream=nullptr; bool on=false;
    GpuSpan(const char* f,int l,const char* op,uint64_t bytes,cudaStream_t st):stream(st) {
        if(!enabled())return;
        if(capturing(st)){add(key(f,l,(std::string(op)+"-capture-template").c_str(),(uintptr_t)st),bytes);return;}
        collect();cudaGetDevice(&p.device);auto& s=gpu_state();
        {std::lock_guard<std::mutex> lock(s.mutex);if(s.pending.size()>=8192){count("gpu-timing-capacity-exceeded");return;}
            auto it=std::find_if(s.free.begin(),s.free.end(),[&](const Pending& x){return x.device==p.device;});
            if(it!=s.free.end()){p=*it;s.free.erase(it);} }
        if(!p.first){if(cudaEventCreate(&p.first)!=cudaSuccess||cudaEventCreate(&p.last)!=cudaSuccess)throw std::runtime_error("transfer audit: event allocation failed");}
        p.k=key(f,l,op,(uintptr_t)st);p.sent=false;
        if(cudaEventRecord(p.first,st)!=cudaSuccess)throw std::runtime_error("transfer audit: timing start failed");on=true;
    }
    ~GpuSpan(){if(!on)return;if(cudaEventRecord(p.last,stream)!=cudaSuccess){count("gpu-end-event-failed");return;}p.sent=true;
        auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);s.pending.push_back(p);}
};
inline cudaError_t copy(const char* f,int l,void* dst,const void* src,size_t n,cudaMemcpyKind kind,cudaStream_t stream) {
    if(!enabled())return cudaMemcpyAsync(dst,src,n,kind,stream);
    const bool cap=capturing(stream);auto op=std::string(direction(kind))+(cap?"-capture-template":"");auto k=key(f,l,op.c_str(),(uintptr_t)stream);
    // Large transfers are timed individually; small calls retain exact byte/API accounting.
    std::unique_ptr<GpuSpan> timing;if(n>=64u<<20&&!cap)timing=std::make_unique<GpuSpan>(f,l,direction(kind),n,stream);
    auto t=Clock::now();auto e=cudaMemcpyAsync(dst,src,n,kind,stream);add(k,n,elapsed(t),e==cudaSuccess);return e;
}
inline cudaError_t copy_sync(const char* f,int l,void* dst,const void* src,size_t n,cudaMemcpyKind kind) {
    if(!enabled())return cudaMemcpy(dst,src,n,kind);auto t=Clock::now();auto e=cudaMemcpy(dst,src,n,kind);add(key(f,l,direction(kind)),n,elapsed(t),e==cudaSuccess);return e;
}
inline cudaError_t copy2d(const char* f,int l,void* dst,size_t dp,const void* src,size_t sp,size_t width,size_t height,cudaMemcpyKind kind) {
    if(!enabled())return cudaMemcpy2D(dst,dp,src,sp,width,height,kind);auto t=Clock::now();auto e=cudaMemcpy2D(dst,dp,src,sp,width,height,kind);add(key(f,l,direction(kind)),width*height,elapsed(t),e==cudaSuccess);return e;
}
inline void release() {
    if(!enabled())return;collect();int d=0;cudaGetDevice(&d);auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);
    for(auto* list:{&s.pending,&s.free})for(auto it=list->begin();it!=list->end();){if(it->device!=d){++it;continue;}cudaEventDestroy(it->first);cudaEventDestroy(it->last);it=list->erase(it);}
    s.streams.erase(std::remove_if(s.streams.begin(),s.streams.end(),[&](auto x){return x.first==d;}),s.streams.end());
}
inline cudaError_t zero(const char* f,int l,void* dst,int value,size_t n,cudaStream_t stream) {
    if(!enabled())return cudaMemsetAsync(dst,value,n,stream);bool cap=capturing(stream);
    std::unique_ptr<GpuSpan> timing;if(n>=64u<<20&&!cap)timing=std::make_unique<GpuSpan>(f,l,"GPU-memset",n,stream);
    auto t=Clock::now();auto e=cudaMemsetAsync(dst,value,n,stream);add(key(f,l,cap?"GPU-memset-capture-template":"GPU-memset",(uintptr_t)stream),n,elapsed(t),e==cudaSuccess);return e;
}
inline cudaError_t sync_stream(const char* f,int l,cudaStream_t stream) {
    if(!enabled())return cudaStreamSynchronize(stream);auto t=Clock::now();auto e=cudaStreamSynchronize(stream);add(key(f,l,"wait-stream",(uintptr_t)stream),0,elapsed(t),e==cudaSuccess);return e;
}
inline cudaError_t sync_event(const char* f,int l,cudaEvent_t event) {
    if(!enabled())return cudaEventSynchronize(event);auto t=Clock::now();auto e=cudaEventSynchronize(event);add(key(f,l,"wait-event",(uintptr_t)event),0,elapsed(t),e==cudaSuccess);return e;
}
inline cudaError_t reg(const char* f,int l,void* p,size_t n,unsigned flags,uint64_t offset) {
    if(!enabled())return cudaHostRegister(p,n,flags);const int before=busy();auto k=key(f,l,"host-register",0,offset);auto t=Clock::now();auto e=cudaHostRegister(p,n,flags);auto ms=elapsed(t);const int after=busy();add(k,n,ms,e==cudaSuccess,before,after);return e;
}
inline cudaError_t unreg(const char* f,int l,void* p,size_t n,uint64_t offset) {
    if(!enabled())return cudaHostUnregister(p);const int before=busy();auto k=key(f,l,"host-unregister",0,offset);auto t=Clock::now();auto e=cudaHostUnregister(p);auto ms=elapsed(t);const int after=busy();add(k,n,ms,e==cudaSuccess,before,after);return e;
}
inline void graph(cudaGraphExec_t exec,cudaGraph_t graph) {
    if(!enabled())return;
#ifndef HIP_VERSION
    size_t n=0;if(cudaGraphGetNodes(graph,nullptr,&n)!=cudaSuccess)throw std::runtime_error("transfer audit: graph nodes unavailable");
    std::vector<cudaGraphNode_t> nodes(n);if(cudaGraphGetNodes(graph,nodes.data(),&n)!=cudaSuccess)throw std::runtime_error("transfer audit: graph nodes unavailable");
    std::vector<GpuState::GraphCopy> copies;
    for(auto node:nodes){cudaGraphNodeType type;if(cudaGraphNodeGetType(node,&type)!=cudaSuccess)continue;
        if(type==cudaGraphNodeTypeMemcpy){cudaMemcpy3DParms p{};if(cudaGraphMemcpyNodeGetParams(node,&p)!=cudaSuccess)continue;
            if(p.srcArray||p.dstArray){count("graph-array-copy-unaccounted");continue;}
            copies.push_back({p.kind,p.extent.width*std::max(size_t(1),p.extent.height)*std::max(size_t(1),p.extent.depth)});}}
    auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);
    auto custom=s.custom_templates.find((uintptr_t)graph);if(custom!=s.custom_templates.end())copies.insert(copies.end(),custom->second.begin(),custom->second.end());
    s.graphs[(uintptr_t)exec]=std::move(copies);
#else
    count("HIP-graph-payload-unaccounted");
#endif
}
inline cudaError_t launch(const char* f,int l,cudaGraphExec_t exec,cudaStream_t stream) {
    if(!enabled())return cudaGraphLaunch(exec,stream);auto t=Clock::now();auto e=cudaGraphLaunch(exec,stream);
    add(key(f,l,"graph-launch",(uintptr_t)stream),0,elapsed(t),e==cudaSuccess);
    if(e==cudaSuccess){auto& s=gpu_state();std::lock_guard<std::mutex> lock(s.mutex);auto it=s.graphs.find((uintptr_t)exec);
        if(it==s.graphs.end())count("graph-payload-unaccounted");else for(auto c:it->second)add(key(f,l,((c.label.empty()?std::string(direction(c.kind)):c.label)+"-graph-replay").c_str(),(uintptr_t)stream),c.bytes);}
    return e;
}
}
#define PA_COPY(...) ::strata::poly::audit::copy(__FILE__,__LINE__,__VA_ARGS__)
#define PA_CREATE_STREAM(...) ::strata::poly::audit::create_stream(__VA_ARGS__)
#define PA_COPY_SYNC(...) ::strata::poly::audit::copy_sync(__FILE__,__LINE__,__VA_ARGS__)
#define PA_COPY2D(...) ::strata::poly::audit::copy2d(__FILE__,__LINE__,__VA_ARGS__)
#define PA_ZERO(...) ::strata::poly::audit::zero(__FILE__,__LINE__,__VA_ARGS__)
#define PA_WAIT_STREAM(...) ::strata::poly::audit::sync_stream(__FILE__,__LINE__,__VA_ARGS__)
#define PA_WAIT_EVENT(...) ::strata::poly::audit::sync_event(__FILE__,__LINE__,__VA_ARGS__)
#define PA_GRAPH_LAUNCH(...) ::strata::poly::audit::launch(__FILE__,__LINE__,__VA_ARGS__)
