// Homework 3
// Daniel Humphreys
// Git repo: https://github.com/Dhumphreys1/tac-hep-gpus/tree/main/week4

#include <stdio.h>
#include <time.h>


#define BLOCK_SIZE 32

const int DSIZE = 256;
const int a = 1;
const int b = 1;

// error checking macro
#define cudaCheckErrors(msg)                                       \
	do {                                                        \
		cudaError_t __err = cudaGetLastError();                 \
		if (__err != cudaSuccess) {                             \
			fprintf(stderr, "Error: %s (%s at %s:%d) \n", msg,           \
			cudaGetErrorString(__err),__FILE__, __LINE__);      \
			fprintf(stderr, "*** FAILED - ABORTING***\n");      \
			exit(1);                                            \
		}                                                       \
	} while (0)


// CUDA kernel that runs on the GPU
__global__ void dot_product(const int *d_A, const int *d_B, int *d_C, int DSIZE) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < DSIZE) {
    atomicAdd(d_C, d_A[idx]*d_B[idx]);
  }
};


int main() {

	// Create the device and host pointers
	int *h_A, *h_B, *h_C, *cpu_result, *d_A, *d_B, *d_C;

	// Fill in the host pointers
  h_A = (int*) malloc(DSIZE*sizeof(int));
  h_B = (int*) malloc(DSIZE*sizeof(int));
  h_C = (int*) malloc(sizeof(int));
  cpu_result = (int*) malloc(sizeof(int));
  *cpu_result = 0;
	for (int i = 0; i < DSIZE; i++){
		h_A[i] = a;
		h_B[i] = b;
    *cpu_result += a*b;
	}

	*h_C = 0;

	// Allocate device memory
  cudaMalloc(&d_A, DSIZE*sizeof(int));
  cudaCheckErrors("Failed to allocate d_A");
  cudaMalloc(&d_B, DSIZE*sizeof(int));
  cudaCheckErrors("Failed to allocate d_B");
  cudaMalloc(&d_C, sizeof(int));
  cudaCheckErrors("Failed to allocate d_C");

	// Copy the matrices on GPU
  cudaMemcpy(d_A, h_A, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaCheckErrors("Failed to copy h_A to d_A");
  cudaMemcpy(d_B, h_B, DSIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaCheckErrors("Failed to copy h_B to d_B");
  cudaMemcpy(d_C, h_C, sizeof(int), cudaMemcpyHostToDevice);
  cudaCheckErrors("Failed to copy h_C to d_C");

	// Define block/grid dimentions and launch kernel
  dim3 blockSize(BLOCK_SIZE);
  dim3 gridSize((DSIZE + blockSize.x - 1) / blockSize.x);

  dot_product<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE);
  cudaCheckErrors("Kernel launch failed");
  cudaDeviceSynchronize();
  cudaCheckErrors("Kernel execution failed");

	// Copy results back to host
  cudaMemcpy(h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost);
  cudaCheckErrors("Failed to copy d_C to h_C");

	// Verify result
  if (*h_C != *cpu_result){
    printf("Cuda result does not match CPU result!\n");
    printf("GPU Result: %d, CPU Result: %d\n", *h_C, *cpu_result);
  }

	// Free allocated memory
  free(h_A);
  free(h_B);
  free(h_C);
  free(cpu_result);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);

	return 0;
}