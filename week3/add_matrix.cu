#include <stdio.h>


const int DSIZE_X = 256;
const int DSIZE_Y = 256;
const int num_elements = DSIZE_X*DSIZE_Y;

__global__ void add_matrix(float *d_A, float *d_B, float *d_C, int DSIZE_X, int DSIZE_Y)
{
    //FIXME:
    // Express in terms of threads and blocks
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    // Add the two matrices - make sure you are not out of range
    if (idx < DSIZE_X && idy < DSIZE_Y){
        int linear_id = idy*DSIZE_X + idx;
        d_C[linear_id] = d_A[linear_id] + d_B[linear_id];
    }
}

int main()
{

    // Create and allocate memory for host and device pointers

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *result;

    h_A = (float*) malloc(num_elements*sizeof(float));
    h_B = (float*) malloc(num_elements*sizeof(float));
    h_C = (float*) malloc(num_elements*sizeof(float));
    result = (float*) malloc(num_elements*sizeof(float));

    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            int idx = j*DSIZE_X + i;
            h_A[idx] = rand()/(float)RAND_MAX;
            h_B[idx] = rand()/(float)RAND_MAX;
            h_C[idx] = 0.0;
            result[idx] = h_A[idx] + h_B[idx];
        }
    }

    cudaMalloc(&d_A, num_elements*sizeof(float));
    cudaMalloc(&d_B, num_elements*sizeof(float));
    cudaMalloc(&d_C, num_elements*sizeof(float));

    cudaMemcpy(d_A, h_A, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, num_elements*sizeof(float), cudaMemcpyHostToDevice);
    // Copy from host to device


    dim3 blockSize(16, 16);
    int grid_dim_x = (DSIZE_X + blockSize.x - 1) / blockSize.x;
    int grid_dim_y = (DSIZE_Y + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(grid_dim_x, grid_dim_y);

    add_matrix<<<gridSize, blockSize>>>(d_A, d_B, d_C, DSIZE_X, DSIZE_Y);

    cudaMemcpy(h_C, d_C, num_elements*sizeof(float), cudaMemcpyDeviceToHost);

    // Print and check some elements to make the addition was successful
    for (int i = 0; i < DSIZE_X; i++) {
        for (int j = 0; j < DSIZE_Y; j++) {
            int idx = j*DSIZE_X + i;
            if (h_C[idx] != result[idx]){
                printf("Error at (%d,%d)\n", i, j);
                return 0;
            }
        }
    }

    printf("Matrix Addition Successful!\n");

    free(h_A);
    free(h_B);
    free(h_C);
    free(result);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}