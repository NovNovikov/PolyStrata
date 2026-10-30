// Standalone real-weight experiments. Only four experts/layer (~1.84 GiB), no GPU/server.
// MODEL.gguf OUTPUT_DIRECTORY [sweeps=8]. Refuses an existing output directory.
#include "cpu_experts.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
using namespace strata;
using namespace strata::poly;
using Clock = std::chrono::steady_clock;
struct LayerData { int id, gu, down, embd, ff; size_t gu_bytes, down_bytes; std::vector<uint8_t> weights; };
struct Variant { const char* name; CpuLayoutProbe options; };
int main(int argc,char** argv) {
    if(argc<3||argc>4)return 2;
    try {
        if(!std::filesystem::create_directory(argv[2]))throw std::runtime_error("output directory exists");
        const auto home=std::filesystem::path(argv[2]);
        const int sweeps=argc==4?std::max(1,std::atoi(argv[3])):8;
        std::ofstream csv(home/"timings.csv"), meta(home/"geometry.csv");
        csv<<"round,rows,variant,ms,prepare_ms,gate_up_ms,quantize_ms,down_ms\n";
        meta<<"layer,gu_type,down_type,embd,ff,weight_bytes\n";
#ifdef _WIN32
        PROCESS_POWER_THROTTLING_STATE off{PROCESS_POWER_THROTTLING_CURRENT_VERSION,0x1ul|0x4ul,0};
        SetProcessInformation(GetCurrentProcess(),ProcessPowerThrottling,&off,sizeof(off));
#endif
        std::vector<LayerData> layers;
        size_t bytes=0;
        {
            GgufFile file(argv[1]);
            for(const auto& gate:file.tensors()) {
                auto at=gate.name.find(".ffn_gate_exps.weight");
                if(at==std::string::npos||gate.shape.size()!=3||gate.shape[2]<4)continue;
                auto prefix=gate.name.substr(0,at);
                auto* up=file.find(prefix+".ffn_up_exps.weight");
                auto* down=file.find(prefix+".ffn_down_exps.weight");
                if(!up||!down||up->type!=gate.type)throw std::runtime_error("expert geometry");
                LayerData l{};l.id=std::stoi(prefix.substr(4));l.gu=gate.type;l.down=down->type;
                l.embd=(int)gate.shape[0];l.ff=(int)gate.shape[1];
                l.gu_bytes=tensor_payload_bytes(gate)/gate.shape[2];l.down_bytes=tensor_payload_bytes(*down)/down->shape[2];
                std::string err;if(!cpu_expert_types_ok(l.gu,l.down,l.embd,l.ff,err))throw std::runtime_error(err);
                l.weights.resize(4*(2*l.gu_bytes+l.down_bytes));
                for(size_t e=0;e<4;++e) {
                    auto* dst=l.weights.data()+e*(2*l.gu_bytes+l.down_bytes);
                    std::memcpy(dst,file.tensor_data(gate)+e*l.gu_bytes,l.gu_bytes);
                    std::memcpy(dst+l.gu_bytes,file.tensor_data(*up)+e*l.gu_bytes,l.gu_bytes);
                    std::memcpy(dst+2*l.gu_bytes,file.tensor_data(*down)+e*l.down_bytes,l.down_bytes);
                }
                meta<<l.id<<','<<l.gu<<','<<l.down<<','<<l.embd<<','<<l.ff<<','<<l.weights.size()<<'\n';
                bytes+=l.weights.size();layers.push_back(std::move(l));
            }
        }
        if(layers.empty())throw std::runtime_error("no layers");
        const std::vector<Variant> variants{
            {"stock",{32,16,false,false}}, {"down_expert_major",{32,16,true,false}},
            {"down_rows64",{32,64,false,false}}, {"down_expert_major64",{32,64,true,false}},
            {"gu_rows16",{16,16,false,false}}, {"gu_rows64",{64,16,false,false}},
            {"gu_sequential",{32,16,false,true}}, {"gu_sequential64",{64,16,false,true}}};
        std::mt19937 rng(1234);
        std::vector<float> x((size_t)8*layers.front().embd),out(x.size()),reference(x.size());
        for(auto& v:x)v=float(int(rng()&1023)-512)/8192.f;
        size_t checks=0;std::atomic<int> callbacks{0};
        auto call=[&](Workers& pool,LayerData& l,int rows,bool callback,bool ragged=false,bool gelu=false){
            std::vector<CpuExpert> jobs;
            for(int row=0;row<rows;++row)for(int e=0;e<(ragged?1+row%4:4);++e) {
                const int which=(e+row)%4;
                auto* p=l.weights.data()+which*(2*l.gu_bytes+l.down_bytes);
                jobs.push_back({p,p+l.gu_bytes,p+2*l.gu_bytes,0.25f,row});
            }
            const std::function<void()> cb=[&]{++callbacks;};
            cpu_experts(pool,jobs.data(),(int)jobs.size(),rows,l.gu,l.gu,l.down,l.embd,l.ff,x.data(),10.f,gelu,
                        out.data(),callback?&cb:nullptr);
        };
        std::printf("loaded %zu layers, %zu bytes; no model/GPU engine\n",layers.size(),bytes);std::fflush(stdout);
        // Ragged rows, reused experts, both activations. Callback must run once.
        for(int threads:{1,12}) {
            Workers pool(threads);
            for(bool gelu:{false,true})for(int rows:{1,2,4,8})for(auto& l:layers) {
                cpu_layout_probe()=variants[0].options;call(pool,l,rows,false,true,gelu);reference=out;
                for(const auto& v:variants) {
                    cpu_layout_probe()=v.options;int before=callbacks.load();call(pool,l,rows,true,true,gelu);
                    if(callbacks!=before+1||std::memcmp(out.data(),reference.data(),(size_t)rows*l.embd*sizeof(float)))
                        throw std::runtime_error(std::string("bitwise/callback mismatch: ")+v.name);
                    ++checks;
                }
            }
        }
        Workers pool(12);
        // Interleave all variants; end each round with a stock control to expose drift.
        for(int round=0;round<5;++round)for(int rows:{1,4}) {
            std::vector<int> order(variants.size());std::iota(order.begin(),order.end(),0);
            std::shuffle(order.begin(),order.end(),rng);order.push_back(0);
            for(int id:order) {
                auto& v=variants[(size_t)id];cpu_layout_probe()=v.options;
                for(auto& l:layers)call(pool,l,rows,false);
                cpu_stage_timing()={};auto start=Clock::now();
                for(int sweep=0;sweep<sweeps;++sweep)for(auto& l:layers)call(pool,l,rows,false);
                double ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count()/sweeps;
                auto t=cpu_stage_timing();
                csv<<round<<','<<rows<<','<<v.name<<','<<ms<<','<<t.prepare/sweeps<<','<<t.gate_up/sweeps<<','
                   <<t.quantize/sweeps<<','<<t.down/sweeps<<'\n';csv.flush();
            }
            std::printf("round=%d rows=%d completed\n",round,rows);std::fflush(stdout);
        }
        std::ofstream done(home/"validation.txt");done<<"bitwise/callback cases="<<checks<<" all passed\n";
        std::printf("checks=%zu passed; complete\n",checks);
    }catch(const std::exception& e){std::fprintf(stderr,"CPU layout probe: %s\n",e.what());return 1;}
}
