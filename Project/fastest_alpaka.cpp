#include <alpaka/alpaka.hpp>
#include <stdio.h>
#include <algorithm>
#include <chrono>

#define RADIUS 3
#define BLOCK_SIZE 32

const int DSIZE = 4096;
const int num_elements = DSIZE*DSIZE;
const float A_val = 3.0f;
const float B_val = 2.0f;

const bool DEBUG = (DSIZE <= 5);

// Alpaka type definitions
using Dim = alpaka::DimInt<2u>;
using Idx = std::size_t;
using Acc = alpaka::AccGpuCudaRt<Dim, Idx>;
using Queue = alpaka::Queue<Acc, alpaka::Blocking>;
using WorkDiv = alpaka::WorkDivMembers<Dim, Idx>;

// Stencil kernel
struct Stencil2DKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                   const float* input,
                                   float* output) const {

        auto& sdata = alpaka::declareSharedVar<float[BLOCK_SIZE + 2*RADIUS][BLOCK_SIZE + 2*RADIUS], __COUNTER__>(acc);

        const int temp_size = (BLOCK_SIZE + 2*RADIUS) * (BLOCK_SIZE + 2*RADIUS);

        const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        const auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

        const int thread_id = threadIdx[0] * blockDim[1] + threadIdx[1];

        // Load shared memory
        for (int i = thread_id; i < temp_size; i += blockDim[0] * blockDim[1]) {
            int temp_x = i % (BLOCK_SIZE + 2*RADIUS);
            int temp_y = i / (BLOCK_SIZE + 2*RADIUS);

            int g_x = blockIdx[1] * BLOCK_SIZE + temp_x - RADIUS;
            int g_y = blockIdx[0] * BLOCK_SIZE + temp_y - RADIUS;

            if ((g_x >= 0) && (g_x < DSIZE) && (g_y >= 0) && (g_y < DSIZE)) {
                sdata[temp_y][temp_x] = input[g_y*DSIZE + g_x];
            } else {
                sdata[temp_y][temp_x] = 0.0f;
            }
        }

        alpaka::syncBlockThreads(acc);

        int x_ind = threadIdx[1] + blockIdx[1] * blockDim[1];
        int y_ind = threadIdx[0] + blockIdx[0] * blockDim[0];
        int x_lindex = threadIdx[1] + RADIUS;
        int y_lindex = threadIdx[0] + RADIUS;

        if ((x_ind >= DSIZE) || (y_ind >= DSIZE)) {
            return;
        }

        float result = 0;
        for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
            result += sdata[y_lindex][x_lindex + offset];
        }
        for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
            result += sdata[y_lindex + offset][x_lindex];
        }
        result -= sdata[y_lindex][x_lindex];

        output[x_ind + DSIZE*y_ind] = result;
    }
};

// Matrix multiplication kernel
struct MatrixMulKernel {
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc,
                                   const float* A,
                                   const float* B,
                                   float* C) const {

        auto& tileA = alpaka::declareSharedVar<float[BLOCK_SIZE][BLOCK_SIZE], __COUNTER__>(acc);
        auto& tileB = alpaka::declareSharedVar<float[BLOCK_SIZE][BLOCK_SIZE], __COUNTER__>(acc);

        const auto threadIdx = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc);
        const auto blockIdx = alpaka::getIdx<alpaka::Grid, alpaka::Blocks>(acc);
        const auto blockDim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc);

        int row = threadIdx[0] + blockDim[0] * blockIdx[0];
        int col = threadIdx[1] + blockDim[1] * blockIdx[1];
        float temp = 0;
        int num_tiles = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

        for(int i = 0; i < num_tiles; i++){
            tileA[threadIdx[0]][threadIdx[1]] = 0.0;
            tileB[threadIdx[0]][threadIdx[1]] = 0.0;

            if((i*BLOCK_SIZE + threadIdx[1]) < DSIZE && row < DSIZE){
                tileA[threadIdx[0]][threadIdx[1]] = A[row*DSIZE + (i*BLOCK_SIZE + threadIdx[1])];
            }
            if((i*BLOCK_SIZE + threadIdx[0]) < DSIZE && col < DSIZE){
                tileB[threadIdx[0]][threadIdx[1]] = B[(i*BLOCK_SIZE + threadIdx[0])*DSIZE + col];
            }

            alpaka::syncBlockThreads(acc);

            for (int j = 0; j < BLOCK_SIZE; j++) {
                float a = tileA[threadIdx[0]][j];
                float b = tileB[j][threadIdx[1]];
                temp += a * b;
            }

            alpaka::syncBlockThreads(acc);
        }

        if (row < DSIZE && col < DSIZE) {
            C[row*DSIZE + col] = temp;
        }
    }
};

void print_matrix(const float *matrix){
    for (int i = 0; i < DSIZE; i++){
        for (int j = 0; j < DSIZE; j++){
            printf("%f ", matrix[i*DSIZE + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void){
    auto start_time = std::chrono::high_resolution_clock::now();

    // Initialize Alpaka
    using Platform = alpaka::Platform<Acc>;
    auto const platform = Platform{};
    auto const devAcc = alpaka::getDevByIdx(platform, 0u);
    Queue queueA(devAcc);
    Queue queueB(devAcc);
    Queue queueMult(devAcc);

    // Host device for CPU buffers
    using DevHost = alpaka::DevCpu;
    using PlatformHost = alpaka::Platform<DevHost>;
    auto const platformHost = PlatformHost{};
    auto const devHost = alpaka::getDevByIdx(platformHost, 0u);

    // Allocate host buffers
    auto h_A = alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(num_elements));
    auto h_B = alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(num_elements));
    auto h_A_stenciled = alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(num_elements));
    auto h_B_stenciled = alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(num_elements));
    auto h_result = alpaka::allocBuf<float, Idx>(devHost, static_cast<Idx>(num_elements));

    // Initialize host buffers
    float* ptrA = alpaka::getPtrNative(h_A);
    float* ptrB = alpaka::getPtrNative(h_B);
    for(int i = 0; i < num_elements; i++){
        ptrA[i] = A_val;
        ptrB[i] = B_val;
    }

    // Device memory
    auto d_A = alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(num_elements));
    auto d_B = alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(num_elements));
    auto d_A_stenciled = alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(num_elements));
    auto d_B_stenciled = alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(num_elements));
    auto d_result = alpaka::allocBuf<float, Idx>(devAcc, static_cast<Idx>(num_elements));

    // Copy to device
    alpaka::memcpy(queueA, d_A, h_A);
    alpaka::memcpy(queueA, d_A_stenciled, h_A_stenciled);
    alpaka::memcpy(queueB, d_B, h_B);
    alpaka::memcpy(queueB, d_B_stenciled, h_B_stenciled);

    // Kernel configuration
    alpaka::Vec<Dim, Idx> const threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    alpaka::Vec<Dim, Idx> const blocksPerGrid(
        (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE
    );
    alpaka::Vec<Dim, Idx> const elementsPerThread(1, 1);
    WorkDiv const workDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

    // Launch stencil kernels
    alpaka::exec<Acc>(queueA, workDiv, Stencil2DKernel{},
                      alpaka::getPtrNative(d_A),
                      alpaka::getPtrNative(d_A_stenciled));
    alpaka::exec<Acc>(queueB, workDiv, Stencil2DKernel{},
                      alpaka::getPtrNative(d_B),
                      alpaka::getPtrNative(d_B_stenciled));

    alpaka::wait(queueA);
    alpaka::wait(queueB);

    // Launch matrix multiply kernel
    alpaka::exec<Acc>(queueMult, workDiv, MatrixMulKernel{},
                      alpaka::getPtrNative(d_A_stenciled),
                      alpaka::getPtrNative(d_B_stenciled),
                      alpaka::getPtrNative(d_result));

    // Copy back
    alpaka::memcpy(queueA, h_A_stenciled, d_A_stenciled);
    alpaka::memcpy(queueB, h_B_stenciled, d_B_stenciled);
    alpaka::memcpy(queueMult, h_result, d_result);

    alpaka::wait(queueA);
    alpaka::wait(queueB);
    alpaka::wait(queueMult);

    if(DEBUG){
        float* ptrAStenciled = alpaka::getPtrNative(h_A_stenciled);
        float* ptrBStenciled = alpaka::getPtrNative(h_B_stenciled);
        float* ptrResult = alpaka::getPtrNative(h_result);
        print_matrix(ptrAStenciled);
        print_matrix(ptrBStenciled);
        print_matrix(ptrResult);
    }

    // Buffers automatically freed when going out of scope

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("Fastest Alpaka total execution time: %ld ms\n", duration.count());

    return 0;
}