#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <chrono>

#define RADIUS 3
#define BLOCK_SIZE 32

const int DSIZE = 2048;
const int num_elements = DSIZE*DSIZE;
const float A_val = 3.0f;
const float B_val = 2.0f;

const bool DEBUG = (DSIZE <= 5);

// error checking macro
#define cudaCheckErrors(msg)                                   \
   do {                                                        \
       cudaError_t __err = cudaGetLastError();                 \
       if (__err != cudaSuccess) {                             \
           fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",  \
                   msg, cudaGetErrorString(__err),             \
                   __FILE__, __LINE__);                        \
           fprintf(stderr, "*** FAILED - ABORTING\n");         \
           exit(1);                                            \
       }                                                       \
   } while (0)

// My original kernels used shared memory. Now I have to write slow ones :(. No copy and paste for me!
__global__ void stencil_2d_slow(const float *input, float *output) {
    int x_ind = threadIdx.x + blockIdx.x * blockDim.x;
    int y_ind = threadIdx.y + blockIdx.y * blockDim.y;
    int g_ind = x_ind + y_ind*DSIZE;
    if ((x_ind >= DSIZE) || (y_ind >= DSIZE)){
        return;
    }

    float result = 0;
    for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
        if ((x_ind + offset >= 0) && (x_ind + offset < DSIZE)){
            result += input[x_ind + offset + y_ind*DSIZE];
        }
        if ((y_ind + offset >= 0) && (y_ind + offset < DSIZE)){
            result += input[x_ind + (y_ind + offset) * DSIZE];
        }
    }
    result -= input[g_ind]; //remove double count
    output[g_ind] = result;
}

__global__ void matrix_mul_slow(const float *A, const float *B, float *C){
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    if ((col >= DSIZE || row >= DSIZE)){
        return;
    }
    float result = 0;
    for (int i = 0; i < DSIZE; i++){
        result += A[row*DSIZE + i]*B[i*DSIZE + col];
    }
    C[row*DSIZE + col] = result;
}
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
	float *A, *B, *A_stenciled, *B_stenciled, *final_result;

	cudaMallocManaged(&A, num_elements*sizeof(float));
    cudaMallocManaged(&B, num_elements*sizeof(float));
    cudaMallocManaged(&A_stenciled, num_elements*sizeof(float));
    cudaMallocManaged(&B_stenciled, num_elements*sizeof(float));
    cudaMallocManaged(&final_result, num_elements*sizeof(float));
    cudaCheckErrors("cudaMallocManaged failure");
	for(int i = 0; i < num_elements; i++){
		A[i] = A_val;
		B[i] = B_val;
		A_stenciled[i] = 0;
		B_stenciled[i] = 0;
		final_result[i] = 0;
	}

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    stencil_2d_slow<<<dimGrid, dimBlock>>>(A, A_stenciled);
    cudaCheckErrors("Stencil A kernel launch failure");
    stencil_2d_slow<<<dimGrid, dimBlock>>>(B, B_stenciled);
    cudaCheckErrors("Stencil B kernel launch failure");
    matrix_mul_slow<<<dimGrid, dimBlock>>>(A_stenciled, B_stenciled, final_result);
    cudaCheckErrors("Matrix multiply kernel launch failure");
    cudaDeviceSynchronize();
    if(DEBUG){
        print_matrix(A_stenciled);
        print_matrix(B_stenciled);
        print_matrix(final_result);
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("Slow total execution time: %ld ms\n", duration.count());
    return 0;
}