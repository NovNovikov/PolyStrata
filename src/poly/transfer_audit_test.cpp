#include "transfer_audit_cuda.hpp"
#include <thread>
#include <stdexcept>
#include <iostream>
using namespace strata::poly;
static void check(bool ok,const char* text){if(!ok)throw std::runtime_error(text);}
static void ck(cudaError_t e){check(e==cudaSuccess,cudaGetErrorString(e));}
int main(){
    check(audit::enabled(),"set STRATA_TRACE_TRANSFERS=1");
    audit::begin_request();audit::set_phase("self-test");
    constexpr size_t n=65536;
    void* host=nullptr;void* device=nullptr;cudaStream_t stream=nullptr;
    ck(cudaMallocHost(&host,n));ck(cudaMalloc(&device,n));ck(audit::create_stream(&stream,cudaStreamNonBlocking));
    PA_MEMSET(host,0x5a,n);
    {audit::GpuSpan timing(__FILE__,__LINE__,"self-test-GPU",n,stream);
     ck(PA_COPY(device,host,n,cudaMemcpyHostToDevice,stream));}
    ck(PA_WAIT_STREAM(stream));
    PA_MEMSET(host,0,n);ck(PA_COPY_SYNC(host,device,n,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<n;++i)check(((unsigned char*)host)[i]==0x5a,"copy payload changed");
    cudaGraph_t g=nullptr;cudaGraphExec_t exec=nullptr;
    ck(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
    ck(PA_COPY(device,host,n,cudaMemcpyHostToDevice,stream));
    ck(cudaStreamEndCapture(stream,&g));ck(cudaGraphInstantiateWithFlags(&exec,g,0));audit::graph(exec,g);
    for(int i=0;i<3;++i)ck(PA_GRAPH_LAUNCH(exec,stream));ck(PA_WAIT_STREAM(stream));audit::collect();
    auto c=audit::context();c.layer=7;
    std::thread worker([&]{audit::Scope scope(c);unsigned char a[128]{},b[128]{};PA_MEMCPY(b,a,sizeof a);});worker.join();
    audit::cache_upload(7,1,128);audit::cache_use(7,1);audit::cache_evict(7,1,"test-evict");
    audit::cache_upload(7,2,128);audit::cache_evict(7,2,"test-evict");
    audit::cache_upload(7,3,128);audit::begin_verify();
    {audit::Scope scope(nullptr,7,-1,2,"verify");audit::cache_use(7,3,0);audit::cache_use(7,3,1);}audit::resolve_verify(1);
    uint64_t h2d=0,replayed=0,templates=0;bool worker_ok=false,gpu_ok=false,unused_ok=false;
    {auto& s=audit::state();std::lock_guard<std::mutex> lock(s.mutex);
     for(auto& item:s.values){const auto& k=item.first;const auto& v=item.second;const auto& op=std::get<7>(k);
       if(op=="H2D")h2d+=v.bytes;if(op=="H2D-graph-replay")replayed+=v.bytes;if(op=="H2D-capture-template")templates+=v.bytes;
       if(op=="CPU-memcpy"&&std::get<3>(k)==7&&v.bytes==128)worker_ok=true;
       if(op=="self-test-GPU"&&v.gpu_samples==1&&v.gpu_ms>0)gpu_ok=true;
       if(op=="cache-evicted-unused"&&v.calls==1)unused_ok=true;}}
    check(h2d==n,"direct H2D count");
#ifndef HIP_VERSION
    check(replayed==3*n,"graph replay count");
#endif
    check(templates==n,"graph templates separated");check(worker_ok,"worker context");check(gpu_ok,"event completion");check(unused_ok,"cache lifetime");
    audit::report("self-test");audit::release();
    ck(cudaGraphExecDestroy(exec));ck(cudaGraphDestroy(g));ck(cudaStreamDestroy(stream));ck(cudaFree(device));ck(cudaFreeHost(host));
    std::cout<<"transfer_audit_test: payloads, graph replay, worker context, GPU timing and cache lifetimes OK\n";
}
