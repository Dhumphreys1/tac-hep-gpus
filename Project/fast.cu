#include <stdio.h>
#include <time.h>
#include <algorithm>
#include <chrono>

#define RADIUS 3
#define BLOCK_SIZE 32

const int DSIZE = 4096;
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

__global__ void stencil_2d_fast(const float *input, float *output) {

	__shared__ float temp[BLOCK_SIZE + 2 * RADIUS][BLOCK_SIZE + 2 * RADIUS];

    // Must devise way to balance the threads loading the halo regions... No reason we can't make it even
    int temp_size = (BLOCK_SIZE + 2 * RADIUS)*(BLOCK_SIZE + 2 * RADIUS);

    // We make all launched threads do work. In the case of a small matrix this is some overhead.
    // However this overhead is dwarfed by memory transfer for matrices this small anyways. Use a cpu for tiny matrices.
    // linear index for the threads in the block
    int thread_id = threadIdx.y * blockDim.x + threadIdx.x;
    // Each thread fills the sahred container at most temp_size // (blockDim.x * blockDim.y) times
    // int( temp_size / (blockSize**2)) = temp_size // (blockSize**2)
    for (int i = thread_id; i < temp_size; i += blockDim.x * blockDim.y){
        int temp_x = i % (BLOCK_SIZE + 2 * RADIUS);
        int temp_y = i / (BLOCK_SIZE + 2 * RADIUS);

        int g_x = blockIdx.x * BLOCK_SIZE + temp_x - RADIUS;
        int g_y = blockIdx.y * BLOCK_SIZE + temp_y - RADIUS;
        // Boundary check
        if ((g_x >= 0) && (g_x < DSIZE) && (g_y >= 0) && (g_y < DSIZE)){
            temp[temp_y][temp_x] = input[g_y*DSIZE + g_x];
        } else {
            temp[temp_y][temp_x] = 0.0f;
        }
    }

    __syncthreads();
    // Now that the shared memory is populated we can do the thing!

    // stencil indices
    int x_ind = threadIdx.x + blockIdx.x * blockDim.x;
	int y_ind = threadIdx.y + blockIdx.y * blockDim.y;
    int x_lindex = threadIdx.x + RADIUS;
	int y_lindex = threadIdx.y + RADIUS;

    // Read input elements into shared memory
	if ((x_ind >= DSIZE) || (y_ind >= DSIZE)){
        return;
    }
	// Apply the stencil
	float result = 0;
	for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
		result += temp[y_lindex][x_lindex + offset];
	}
	for (int offset = -RADIUS; offset <= RADIUS; ++offset) {
		result += temp[y_lindex + offset][x_lindex];
	}
    result -= temp[y_lindex][x_lindex]; //remove double count
	// Store the result
	output[x_ind + DSIZE*y_ind] = result;
}

__global__ void matrix_mul_fast(const float *A, const float *B, float *C) {
    __shared__ float tileA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tileB[BLOCK_SIZE][BLOCK_SIZE];

    int row = threadIdx.y + blockDim.y * blockIdx.y;
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    float temp = 0;
    int num_tiles = (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for(int i = 0; i < num_tiles; i++){
        tileA[threadIdx.y][threadIdx.x] = 0.0;
        tileB[threadIdx.y][threadIdx.x] = 0.0;

        if((i*BLOCK_SIZE + threadIdx.x) < DSIZE && row < DSIZE){
            tileA[threadIdx.y][threadIdx.x] = A[row*DSIZE + (i*BLOCK_SIZE + threadIdx.x)];
        }
        if((i*BLOCK_SIZE + threadIdx.y) < DSIZE && col < DSIZE){
            tileB[threadIdx.y][threadIdx.x] = B[(i*BLOCK_SIZE + threadIdx.y)*DSIZE + col];
        }

        __syncthreads();

        for(int j = 0; j < BLOCK_SIZE; j++){
            temp += tileA[threadIdx.y][j] * tileB[j][threadIdx.x];
        }
    }

    if (row < DSIZE && col < DSIZE) {
        C[row*DSIZE + col] = temp;
    }
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
    // We can use up to 3 streams to optimize the memory transfers and operations
    // The matrix multiplication will have to wait for the stencil streams to finish however.
    cudaStream_t stencil_A, stencil_B, matrix_mult;
    cudaStreamCreate(&stencil_A);
    cudaCheckErrors("Failed to create stencil_A");
    cudaStreamCreate(&stencil_B);
    cudaCheckErrors("Failed to create stencil_B");
    cudaStreamCreate(&matrix_mult);
    cudaCheckErrors("Failed to create matrix_mult");

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

    // To utilize the streams we must use the async memory copy operations
    cudaMemcpyAsync(d_A, h_A, num_elements*sizeof(float), cudaMemcpyHostToDevice, stencil_A);
    cudaCheckErrors("Failed to copy h_A to d_A");
    cudaMemcpyAsync(d_A_stenciled, h_A_stenciled, num_elements*sizeof(float), cudaMemcpyHostToDevice, stencil_A);
    cudaCheckErrors("Failed to copy h_A_stenciled to d_A_stenciled");

    cudaMemcpyAsync(d_B, h_B, num_elements*sizeof(float), cudaMemcpyHostToDevice, stencil_B);
    cudaCheckErrors("Failed to copy h_B to d_B");
    cudaMemcpyAsync(d_B_stenciled, h_B_stenciled, num_elements*sizeof(float), cudaMemcpyHostToDevice, stencil_B);
    cudaCheckErrors("Failed to copy h_B_stenciled to d_B_stenciled");

    cudaMemcpyAsync(d_result, h_result, num_elements*sizeof(float), cudaMemcpyHostToDevice, matrix_mult);
    cudaCheckErrors("Failed to copy h_result to d_result");


    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, (DSIZE + BLOCK_SIZE - 1) / BLOCK_SIZE);
    stencil_2d_fast<<<dimGrid, dimBlock, 0, stencil_A>>>(d_A, d_A_stenciled);
    cudaCheckErrors("Stencil A kernel launch failure");
    stencil_2d_fast<<<dimGrid, dimBlock, 0, stencil_B>>>(d_B, d_B_stenciled);
    cudaCheckErrors("Stencil B kernel launch failure");

    cudaStreamSynchronize(stencil_A);
    cudaStreamSynchronize(stencil_B);

    matrix_mul_fast<<<dimGrid, dimBlock, 0, matrix_mult>>>(d_A_stenciled, d_B_stenciled, d_result);
    cudaCheckErrors("Matrix multiply kernel launch failure");

    // Now copy back asynchronously in parallel streams
    cudaMemcpyAsync(h_A_stenciled, d_A_stenciled, num_elements*sizeof(float), cudaMemcpyDeviceToHost, stencil_A);
    cudaCheckErrors("Failed to copy d_A_stenciled to h_A_stenciled");
    cudaMemcpyAsync(h_B_stenciled, d_B_stenciled, num_elements*sizeof(float), cudaMemcpyDeviceToHost, stencil_B);
    cudaCheckErrors("Failed to copy d_B_stenciled to h_B_stenciled");
    cudaMemcpyAsync(h_result, d_result, num_elements*sizeof(float), cudaMemcpyDeviceToHost, matrix_mult);
    cudaCheckErrors("Failed to copy d_result to h_result");

    // Wait for all async copies to complete
    cudaStreamSynchronize(stencil_A);
    cudaStreamSynchronize(stencil_B);
    cudaStreamSynchronize(matrix_mult);

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
    printf("Fast total execution time: %ld ms\n", duration.count());
    return 0;
}