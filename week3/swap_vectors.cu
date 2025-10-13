#include <stdio.h>


const int DSIZE = 40960;
const int block_size = 256;
const int grid_size = DSIZE/block_size;


__global__ void swap_vectors (float *d_A, float *d_B, int vsize) {

    //FIXME:
    // Express the vector index in terms of threads and blocks
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < vsize){
        float temp = d_A[idx];
        d_A[idx] = d_B[idx];
        d_B[idx] = temp;
    }

    // Swap the vector elements - make sure you are not out of range

}


int main() {


    float *h_A, *h_B, *d_A, *d_B, *orig_A, *orig_B;
    h_A = new float[DSIZE];
    h_B = new float[DSIZE];
    orig_A = new float[DSIZE];
    orig_B = new float[DSIZE];


    for (int i = 0; i < DSIZE; i++) {
        h_A[i] = rand()/(float)RAND_MAX;
        orig_A[i] = h_A[i];
        h_B[i] = rand()/(float)RAND_MAX;
        orig_B[i] = h_B[i];
    }

    cudaMalloc(&d_A, DSIZE*sizeof(float));
    cudaMalloc(&d_B, DSIZE*sizeof(float));

    cudaMemcpy(d_A, h_A, DSIZE*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, DSIZE*sizeof(float), cudaMemcpyHostToDevice);

    swap_vectors<<<grid_size, block_size>>>(d_A, d_B, DSIZE);

    cudaMemcpy(h_A, d_A, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_B, d_B, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < DSIZE; i++){
        if (h_A[i] != orig_B[i] || h_B[i] != orig_A[i]){
            printf("Error at index %d\n", i);
            break;
        }
    }
    printf("Swap Verified!\n");

    delete[] h_A;
    delete[] h_B;
    delete[] orig_A;
    delete[] orig_B;
    cudaFree(d_A);
    cudaFree(d_B);

    return 0;
}
