// External graph events must notify CPU routing on every replay, not a previous launch.
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>

static void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void pause_gpu(unsigned long long cycles) {
    const auto start = clock64();
    while (clock64() - start < cycles) {}
}
int main() {
    int* host = nullptr;
    int* device = nullptr;
    cudaStream_t stream = nullptr;
    cudaEvent_t event = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    cudaGraph_t overlap = nullptr;
    cudaGraphExec_t overlap_exec = nullptr;
    int result = 1;
    try {
        check(cudaMallocHost(&host, 2 * sizeof(int)));
        host[0] = -11; host[1] = -22;
        check(cudaMalloc(&device, sizeof(int)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        const auto capture = [&](unsigned long long tail, cudaGraph_t& source, cudaGraphExec_t& ready) {
            check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
            check(cudaMemcpyAsync(device, host, sizeof(int), cudaMemcpyHostToDevice, stream));
            pause_gpu<<<1,1,0,stream>>>(128000);
            check(cudaMemcpyAsync(host + 1, device, sizeof(int), cudaMemcpyDeviceToHost, stream));
            check(cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal));
            pause_gpu<<<1,1,0,stream>>>(tail); // shared work behind the router event
            check(cudaStreamEndCapture(stream, &source));
            check(cudaGraphInstantiateWithFlags(&ready, source, 0));
        };
        capture(128000, graph, exec);
        check(cudaGraphUpload(exec, stream));
        check(cudaStreamSynchronize(stream));
        if (host[1] != -22) throw std::runtime_error("graph upload executed inference work");
        for (int i = 1; i <= 1000; ++i) {
            host[0] = i; host[1] = -1;
            check(cudaGraphLaunch(exec, stream));
            check(cudaEventSynchronize(event));
            if (host[1] != i) throw std::runtime_error("external event exposed a stale router buffer");
        }
        check(cudaStreamSynchronize(stream));
        capture(256000000, overlap, overlap_exec);
        host[0] = 12345; host[1] = -1;
        check(cudaGraphLaunch(overlap_exec, stream));
        cudaError_t observed;
        do { observed = cudaEventQuery(event); } while (observed == cudaErrorNotReady);
        check(observed);
        if (host[1] != 12345) throw std::runtime_error("overlap probe exposed stale router data");
        if (cudaStreamQuery(stream) != cudaErrorNotReady)
            throw std::runtime_error("router event was not observable before trailing GPU work finished");
        check(cudaStreamSynchronize(stream));
        std::puts("PASS: upload executed no work; 1000 current-data replays; host router event precedes trailing GPU work");
        result = 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
    }
    if (stream) cudaStreamSynchronize(stream);
    if (overlap_exec) cudaGraphExecDestroy(overlap_exec);
    if (overlap) cudaGraphDestroy(overlap);
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    if (event) cudaEventDestroy(event);
    if (stream) cudaStreamDestroy(stream);
    if (device) cudaFree(device);
    if (host) cudaFreeHost(host);
    return result;
}
