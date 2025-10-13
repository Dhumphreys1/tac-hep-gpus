#include <stdio.h>
#include <time.h>

//Increased DSIZE to see gpu speed bottleneck
//This verified the bottleneck was entirely due to data transferring from host to device.
//Matrix so small my shared_col trick didn't do anything xD
const int DSIZE = 512;
const int num_elements = DSIZE*DSIZE;
const int shared_col_size_max = 256;
const float A_val = 3.0f;
const float B_val = 2.0f;

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

// Square matrix multiplication on CPU : C = A * B
void matrix_mul_cpu(const float *A, const float *B, float *C, int size) {
  for (int row = 0; row < size; row++){
    for (int col = 0; col < size; col++){
      int c_id = row*size + col;
      C[c_id] = 0;
      for (int i = 0; i < size; i++){
          C[c_id] += A[i + row*size]*B[i*size + col];
      }
    }
  }
}

// Square matrix multiplication on GPU : C = A * B
__global__ void matrix_mul_gpu(const float *A, const float *B, float *C, int size) {
  __shared__ float sharedBcol[shared_col_size_max]; // fixed to be 256, in case we put a HUGE matrix
    int col = threadIdx.x + blockDim.x * blockIdx.x;
    int row = threadIdx.y + blockDim.y * blockIdx.y;
    // Make sure we are not out of range
    if ((col < size) && (row < size)) {
      float temp = 0.0;
      // compute partial sums of shared_cols
      for (int k = 0; k < size; k += shared_col_size_max){
        int shared_col_size = min(shared_col_size_max, size - k);
        // load in shared cols.
        // Since asked to use blocks of m by m we have to split the work in this fashion
        // This 'bug' took me a while. Its cleaner with 1 dim blocks.
        for (int i = threadIdx.y; i < shared_col_size; i += blockDim.y) {
            sharedBcol[i] = B[(k + i)*size + col];
        }
        //sync threads to make sure they have the same sharedBcol
        __syncthreads();

        //dot product of A with sharedBcol
        for (int n = 0; n < shared_col_size; n++){
          temp += A[row*size + n + k]*sharedBcol[n];
        }
        //sync again before rewriting shared memory
        __syncthreads();
      }
      // partial sums finished assign C value.
      C[row*size+col] = temp;
    }

}

int main() {

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *cpu_result;

    // These are used for timing
    clock_t t0, t1, t2, t3;
    double t1sum=0.0;
    double t2sum=0.0;
    double t3sum=0.0;

    // start timing
    t0 = clock();

    // N*N matrices defined in 1 dimention
    // If you prefer to do this in 2-dimentions cupdate accordingly
    // if using new keyword then you must use delete to free the memory.
    // Gonna get used to the C syntax instead.
    h_A = (float*) malloc(num_elements*sizeof(float));
    h_B = (float*) malloc(num_elements*sizeof(float));
    h_C = (float*) malloc(num_elements*sizeof(float));
    cpu_result = (float*) malloc(num_elements*sizeof(float));
    for (int i = 0; i < num_elements; i++){
        h_A[i] = A_val;
        h_B[i] = B_val;
        h_C[i] = 0;
        cpu_result[i] = 0;
    }

    // Initialization timing
    t1 = clock();
    t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
    printf("Init took %f seconds.  Begin compute\n", t1sum);

    // Allocate device memory and copy input data from host to device
    cudaMalloc(&d_A, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_A");
    cudaMalloc(&d_B, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_B");
    cudaMalloc(&d_C, num_elements*sizeof(float));
    cudaCheckErrors("Failed to allocate d_C");

    cudaMemcpy(d_A, h_A, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_A to d_A");
    cudaMemcpy(d_B, h_B, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_B to d_B");
    cudaMemcpy(d_C, h_C, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("Failed to copy h_C to d_C");


    dim3 blockSize(16, 16);
    int grid_dim_x = (DSIZE+ blockSize.x - 1) / blockSize.x;
    int grid_dim_y = (DSIZE + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(grid_dim_x, grid_dim_y);
    matrix_mul_gpu<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE);
    cudaCheckErrors("Kernel launch failed");
    cudaDeviceSynchronize();
    cudaCheckErrors("Kernel execution failed");

    // Copy results back to host
    cudaMemcpy(h_C, d_C, num_elements*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("Failed to copy d_C to h_C");


    // GPU timing
    t2 = clock();
    t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t2sum);

    matrix_mul_cpu(h_A, h_B, cpu_result, DSIZE);

    // CPU timing
    t3 = clock();
    t3sum = ((double)(t3-t2))/CLOCKS_PER_SEC;
    printf ("Done. Compute took %f seconds\n", t3sum);

    for (int i = 0; i < num_elements; i++){
      if(cpu_result[i] != h_C[i]){
        printf("Result Mismatch! %d != %d\n",cpu_result[i], h_C[i]);
        free(h_A);
        free(h_B);
        free(h_C);
        free(cpu_result);
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return 0;
      }
    }

    printf("Multiplication 'Verified'!\n");
    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(cpu_result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;

}
