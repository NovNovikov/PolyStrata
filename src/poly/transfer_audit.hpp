// Opt-in numeric accounting. Never records prompts, token IDs, tensor contents or session names.
#pragma once
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include <cstdint>
#include <algorithm>

namespace strata::poly::audit {
inline bool enabled() { static const bool on = [] { const char* p = std::getenv("STRATA_TRACE_TRANSFERS"); return p && *p && std::strcmp(p,"0"); }(); return on; }
using Clock = std::chrono::steady_clock;
inline double elapsed(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
struct Context { uint64_t request=0; const char* phase="load"; int layer=-1, pass=0, rows=0; const char* work="ordinary"; };
inline std::atomic<uint64_t> request{0};
inline std::atomic<const char*> phase{"load"};
inline thread_local Context local;
inline Context context() { auto c=local; if (!c.request) { c.request=request.load(); c.phase=phase.load(); } return c; }
struct Scope {
    Context before; bool on;
    explicit Scope(Context c):on(enabled()) { if(on) { before=local; local=c; } }
    Scope(const char* p, int layer=-2, int pass=-1, int rows=-1, const char* work=nullptr):on(enabled()) {
        if(on) { before=local; local=context(); if(p) local.phase=p; if(layer!=-2)local.layer=layer;
            if(pass>=0)local.pass=pass; if(rows>=0)local.rows=rows; if(work)local.work=work; }
    }
    ~Scope(){ if(on)local=before; }
};
// Request, phase, work, layer, pass, call site, stream, detail (e.g. arena offset).
using Key=std::tuple<uint64_t,std::string,std::string,int,int,std::string,int,std::string,uintptr_t,uint64_t>;
struct Value { uint64_t calls=0,bytes=0,failed=0,busy_before=0,busy_after=0,busy_cleared=0; double host_ms=0,max_ms=0,gpu_ms=0,host_busy_before_ms=0; uint64_t gpu_samples=0; };
struct Life { bool active=false; uint64_t uses=0,generation=0; Clock::time_point uploaded{}; };
struct State { std::mutex mutex; std::map<Key,Value> values; std::map<uint64_t,Life> cache; std::map<uint64_t,uint64_t> previous_partner;
    uint64_t blob_live=0,blob_peak=0; std::vector<int> verify_uses; };
inline State& state() { static State s; return s; }
inline Key key(const char* file,int line,const char* op,uintptr_t stream=0,uint64_t detail=0) {
    const auto c=context(); const char* name=file; for(const char* p=file;*p;++p)if(*p=='/'||*p=='\\')name=p+1;
    return {c.request,c.phase,c.work,c.layer,c.pass,name,line,op,stream,detail};
}
inline void add(const Key& k,uint64_t bytes,double ms=0,bool success=true,int busy_before=0,int busy_after=0) {
    auto& s=state(); std::lock_guard<std::mutex> lock(s.mutex); auto& v=s.values[k];
    ++v.calls; if(success)v.bytes+=bytes; else ++v.failed; v.host_ms+=ms; v.max_ms=std::max(v.max_ms,ms);
    v.busy_before+=busy_before>0; v.busy_after+=busy_after>0;v.busy_cleared+=busy_before>0&&busy_after==0;if(busy_before>0)v.host_busy_before_ms+=ms;
}
inline void count(const char* op,uint64_t bytes=0,uint64_t detail=0) { if(enabled())add(key("semantic",0,op,0,detail),bytes); }
inline void blob_memory(size_t bytes,bool allocate) { if(!enabled())return;auto& s=state();
    {std::lock_guard<std::mutex> lock(s.mutex);if(allocate)s.blob_live+=bytes;else s.blob_live-=bytes;s.blob_peak=std::max(s.blob_peak,s.blob_live);}
    count(allocate?"session-blob-allocate":"session-blob-free",bytes); }
inline void begin_verify() {if(enabled()){auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);s.verify_uses.clear();}}
inline void resolve_verify(int accepted) {if(!enabled())return;auto& s=state();uint64_t yes=0,no=0;
    {std::lock_guard<std::mutex> lock(s.mutex);for(int row:s.verify_uses)(row<accepted?yes:no)++;s.verify_uses.clear();}
    if(yes)add(key("semantic",0,"cache-verify-accepted-uses"),0,0,true);if(no)add(key("semantic",0,"cache-verify-rejected-uses"),0,0,true);
    {std::lock_guard<std::mutex> lock(s.mutex);if(yes)s.values[key("semantic",0,"cache-verify-accepted-uses")].calls+=yes-1;if(no)s.values[key("semantic",0,"cache-verify-rejected-uses")].calls+=no-1;}}
inline void gpu(const Key& k,double ms) { auto& s=state(); std::lock_guard<std::mutex> lock(s.mutex); auto& v=s.values[k];v.gpu_ms+=ms;++v.gpu_samples; }
inline void set_phase(const char* p) { if(enabled()){phase.store(p);local.phase=p;} }
inline void begin_request() { if(enabled()){local={};local.request=++request;local.phase="restore";phase.store("restore");} }
inline void* memcpy(const char* f,int l,void* d,const void* s,size_t n) {
    if(!enabled())return std::memcpy(d,s,n); auto k=key(f,l,"CPU-memcpy");auto t=Clock::now();auto r=std::memcpy(d,s,n);add(k,n,elapsed(t));return r;
}
inline void* memset(const char* f,int l,void* d,int value,size_t n) {
    if(!enabled())return std::memset(d,value,n);auto k=key(f,l,"CPU-memset");auto t=Clock::now();auto r=std::memset(d,value,n);add(k,n,elapsed(t));return r;
}
inline size_t fread(const char* f,int l,void* p,size_t n,size_t m,FILE* h) {
    if(!enabled())return std::fread(p,n,m,h);auto k=key(f,l,"file-read-buffered");auto t=Clock::now();auto r=std::fread(p,n,m,h);add(k,n*r,elapsed(t),r==m);return r;
}
inline size_t fwrite(const char* f,int l,const void* p,size_t n,size_t m,FILE* h) {
    if(!enabled())return std::fwrite(p,n,m,h);auto k=key(f,l,"file-write-buffered");auto t=Clock::now();auto r=std::fwrite(p,n,m,h);add(k,n*r,elapsed(t),r==m);return r;
}
struct HostSpan {
    bool on; Key k; Clock::time_point t; uint64_t bytes;
    HostSpan(const char* f,int line,const char* op,uint64_t n=0):on(enabled()),bytes(n) { if(on){k=key(f,line,op);t=Clock::now();} }
    ~HostSpan(){if(on)add(k,bytes,elapsed(t));}
};
inline uint64_t expert_id(int layer,int e){return (uint64_t(uint32_t(layer))<<32)|uint32_t(e);}
inline void cache_evict(int layer,int e,const char* reason) {
    if(!enabled())return;auto& s=state();bool unused=false,active=false;uint64_t uses=0;
    {std::lock_guard<std::mutex> lock(s.mutex);auto& v=s.cache[expert_id(layer,e)];active=v.active;unused=!v.uses;uses=v.uses;v.active=false;}
    if(active){count(reason,0,expert_id(layer,e));count(unused?"cache-evicted-unused":"cache-evicted-used",0,expert_id(layer,e));}
}
inline void cache_upload(int layer,int e,uint64_t bytes) {
    if(!enabled())return;auto& s=state();bool repeat;
    {std::lock_guard<std::mutex> lock(s.mutex);auto& v=s.cache[expert_id(layer,e)];repeat=v.generation>0;v.active=true;v.uses=0;++v.generation;v.uploaded=Clock::now();}
    count(repeat?"cache-reupload":"cache-first-upload",bytes,expert_id(layer,e));
}
inline void cache_use(int layer,int e,int row=0) {
    if(!enabled())return;auto& s=state();double delay=0;bool first=false,active=false;
    {std::lock_guard<std::mutex> lock(s.mutex);auto i=s.cache.find(expert_id(layer,e));if(i!=s.cache.end()&&i->second.active){active=true;first=i->second.uses++==0;if(first)delay=elapsed(i->second.uploaded);if(std::strcmp(context().work,"verify")==0)s.verify_uses.push_back(row);}}
    if(active){count("cache-use",0,expert_id(layer,e));if(first)add(key("semantic",0,"cache-first-use-delay",0,expert_id(layer,e)),0,delay);}
}
inline void exchange(int layer,int hot,int cold,uint64_t bytes) {
    if(!enabled())return;auto& s=state();bool back=false;auto h=expert_id(layer,hot),c=expert_id(layer,cold);
    {std::lock_guard<std::mutex> lock(s.mutex);auto i=s.previous_partner.find(h);back=i!=s.previous_partner.end()&&i->second==c;s.previous_partner[c]=h;}
    cache_evict(layer,cold,"exchange-demotion");cache_upload(layer,hot,bytes);count(back?"exchange-ping-pong":"exchange-new-pair",bytes);
}
inline std::string escaped(const std::string& in){std::string out;for(char c:in){if(c=='\\'||c=='"')out+='\\';if(c=='\n'){out+="\\n";continue;}out+=c;}return out;}
inline void report(const char* boundary) {
    if(!enabled())return;auto& s=state();std::lock_guard<std::mutex> lock(s.mutex);
    for(const auto& item:s.values){const auto& k=item.first;const auto& v=item.second;
        std::fprintf(stderr,"poly-audit {\"boundary\":\"%s\",\"request\":%llu,\"phase\":\"%s\",\"work\":\"%s\",\"layer\":%d,\"pass\":%d,\"file\":\"%s\",\"line\":%d,\"op\":\"%s\",\"stream\":%llu,\"detail\":%llu,\"calls\":%llu,\"bytes\":%llu,\"failed\":%llu,\"host_ms\":%.6f,\"max_ms\":%.6f,\"gpu_ms\":%.6f,\"gpu_samples\":%llu,\"busy_before\":%llu,\"busy_after\":%llu,\"busy_cleared\":%llu,\"host_busy_before_ms\":%.6f}\n",
        boundary,(unsigned long long)std::get<0>(k),escaped(std::get<1>(k)).c_str(),escaped(std::get<2>(k)).c_str(),std::get<3>(k),std::get<4>(k),escaped(std::get<5>(k)).c_str(),std::get<6>(k),escaped(std::get<7>(k)).c_str(),(unsigned long long)std::get<8>(k),(unsigned long long)std::get<9>(k),(unsigned long long)v.calls,(unsigned long long)v.bytes,(unsigned long long)v.failed,v.host_ms,v.max_ms,v.gpu_ms,(unsigned long long)v.gpu_samples,(unsigned long long)v.busy_before,(unsigned long long)v.busy_after,(unsigned long long)v.busy_cleared,v.host_busy_before_ms);
    }s.values.clear();
    std::fprintf(stderr,"poly-audit {\"boundary\":\"%s\",\"request\":%llu,\"op\":\"session-blob-gauge\",\"live\":%llu,\"peak\":%llu}\n",boundary,(unsigned long long)request.load(),(unsigned long long)s.blob_live,(unsigned long long)s.blob_peak);
    s.blob_peak=s.blob_live;
}
}
#define PA_MEMCPY(...) ::strata::poly::audit::memcpy(__FILE__,__LINE__,__VA_ARGS__)
#define PA_MEMSET(...) ::strata::poly::audit::memset(__FILE__,__LINE__,__VA_ARGS__)
#define PA_FREAD(...) ::strata::poly::audit::fread(__FILE__,__LINE__,__VA_ARGS__)
#define PA_FWRITE(...) ::strata::poly::audit::fwrite(__FILE__,__LINE__,__VA_ARGS__)
