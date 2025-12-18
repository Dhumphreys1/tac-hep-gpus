#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <chrono>

#define N 64
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
	float *h_A, *h_B, *h_A_stenciled, *h_B_stenciled, *h_result;
    float *d_A, *d_B, *d_A_stenciled, *d_B_stenciled, *d_result;
	h_A = (float*) malloc(num_elements*sizeof(float));
	h_B = (float*) malloc(num_elements*sizeof(float));
	h_A_stenciled = (float*) malloc(num_elements*sizeof(float));
	h_B_stenciled = (float*) malloc(num_elements*sizeof(float));
	//h_C = (float*) malloc(num_elements*sizeof(float));
	h_result = (float*) malloc(num_elements*sizeof(float));
	for(int i = 0; i < num_elements; i++){
		h_A[i] = A_val;
		h_B[i] = B_val;
		h_A_stenciled[i] = 0;
		h_B_stenciled[i] = 0;
		//h_C[i] = 0;
		h_result[i] = 0;
	}
    cudaMalloc(&d_A, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_A");
    cudaMalloc(&d_A_stenciled, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_A_stenciled");
    cudaMalloc(&d_B, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_B");
    cudaMalloc(&d_B_stenciled, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_B_stenciled");
    cudaMalloc(&d_result, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_C");

    cudaMemcpy(d_A, h_A, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_A to d_A");
    cudaMemcpy(d_A_stenciled, h_A_stenciled, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_A_stenciled to d_A_stenciled");
    cudaMemcpy(d_B, h_B, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_B to d_B");
    cudaMemcpy(d_B_stenciled, h_B_stenciled, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_B_stenciled to d_B_stenciled");
    cudaMemcpy(d_result, h_result, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_result to d_result");


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    stencil_2d_slow<<<dimGrid, dimBlock>>>(d_A, d_A_stenciled);
    cudaCheckErrors("Stencil A kernel launch failure");
    stencil_2d_slow<<<dimGrid, dimBlock>>>(d_B, d_B_stenciled);
    cudaCheckErrors("Stencil B kernel launch failure");
    matrix_mul_slow<<<dimGrid, dimBlock>>>(d_A_stenciled, d_B_stenciled, d_result);
    cudaCheckErrors("Matrix multiply kernel launch failure");

    cudaDeviceSynchronize();

    cudaMemcpy(h_A_stenciled, d_A_stenciled, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy d_A_stenciled to h_A_stenciled");
    cudaMemcpy(h_B_stenciled, d_B_stenciled, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy d_B_stenciled to h_B_stenciled");
    cudaMemcpy(h_result, d_result, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy d_result to h_result");
    if(DEBUG){
        print_matrix(h_A_stenciled);
        print_matrix(h_B_stenciled);
        print_matrix(h_result);
    }
    free(h_A);
    free(h_B);
    free(h_A_stenciled);
    free(h_B_stenciled);
    free(h_result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_result);

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    printf("Average total execution time: %ld ms\n", duration.count());
    return 0;
}