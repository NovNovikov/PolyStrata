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
    int result = 1;
    try {
        check(cudaMallocHost(&host, 2 * sizeof(int)));
        check(cudaMalloc(&device, sizeof(int)));
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        check(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        check(cudaMemcpyAsync(device, host, sizeof(int), cudaMemcpyHostToDevice, stream));
        pause_gpu<<<1,1,0,stream>>>(128000);
        check(cudaMemcpyAsync(host + 1, device, sizeof(int), cudaMemcpyDeviceToHost, stream));
        check(cudaEventRecordWithFlags(event, stream, cudaEventRecordExternal));
        pause_gpu<<<1,1,0,stream>>>(128000); // shared-expert work is still allowed behind the router event
        check(cudaStreamEndCapture(stream, &graph));
        check(cudaGraphInstantiateWithFlags(&exec, graph, 0));
        for (int i = 1; i <= 1000; ++i) {
            host[0] = i; host[1] = -1;
            check(cudaGraphLaunch(exec, stream));
            check(cudaEventSynchronize(event));
            if (host[1] != i) throw std::runtime_error("external event exposed a stale router buffer");
        }
        check(cudaStreamSynchronize(stream));
        std::puts("PASS: 1000 graph replays expose the current host router data through the external event");
        result = 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
    }
    if (stream) cudaStreamSynchronize(stream);
    if (exec) cudaGraphExecDestroy(exec);
    if (graph) cudaGraphDestroy(graph);
    if (event) cudaEventDestroy(event);
    if (stream) cudaStreamDestroy(stream);
    if (device) cudaFree(device);
    if (host) cudaFreeHost(host);
    return result;
}
