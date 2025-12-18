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

// Preforms stencil operation on a matrix
void stencil_2d(const float *input, float *output){
	// Loop through all elements of the flat matrix
	for(int x_ind = 0; x_ind < DSIZE; x_ind++){
		for(int y_ind = 0; y_ind < DSIZE; y_ind++){
			int g_ind = x_ind + y_ind*DSIZE;
			for(int offset = -RADIUS; offset<=RADIUS; offset++){
				// Check that the x and y index are valid indices for the edges of the stencil
				// this way we have 1 offset loop rather than two.
				if ((x_ind + offset >= 0) && (x_ind + offset < DSIZE)){
								output[g_ind] += input[x_ind + offset + y_ind*DSIZE];
				}
				if ((y_ind + offset >= 0) && (y_ind + offset < DSIZE)){
								output[g_ind] += input[x_ind + (y_ind + offset) * DSIZE];
				}
			}
			output[g_ind] -= input[g_ind]; // remove the center element which was added twice
		}
	}
}

void matrix_mult(const float *A, const float *B, float *C){
	for(int a_row = 0; a_row < DSIZE; a_row++){
		for(int shared_dim = 0; shared_dim < DSIZE; shared_dim++){
			float a_val = A[a_row*DSIZE + shared_dim];
			for(int b_col = 0; b_col < DSIZE; b_col++){
				C[a_row*DSIZE + b_col] += a_val * B[shared_dim*DSIZE + b_col];
			}
		}
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
	float *h_A, *h_B, *h_A_stenciled, *h_B_stenciled, *final_result;
	h_A = (float*) malloc(num_elements*sizeof(float));
	h_B = (float*) malloc(num_elements*sizeof(float));
	h_A_stenciled = (float*) malloc(num_elements*sizeof(float));
	h_B_stenciled = (float*) malloc(num_elements*sizeof(float));
	//h_C = (float*) malloc(num_elements*sizeof(float));
	final_result = (float*) malloc(num_elements*sizeof(float));
	for(int i = 0; i < num_elements; i++){
		h_A[i] = A_val;
		h_B[i] = B_val;
		h_A_stenciled[i] = 0;
		h_B_stenciled[i] = 0;
		//h_C[i] = 0;
		final_result[i] = 0;
	}

	stencil_2d(h_A, h_A_stenciled);
	stencil_2d(h_B, h_B_stenciled);
	matrix_mult(h_A_stenciled, h_B_stenciled, final_result);
	if(DEBUG){
		// set DSIZE to a small number, 4 or less to verify correctness
		printf("Matrix A after stencil:\n");
		print_matrix(h_A_stenciled);
		printf("Matrix B after stencil:\n");
		print_matrix(h_B_stenciled);
		printf("Final Result:\n");
		print_matrix(final_result);
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
	printf("CPU total execution time: %ld ms\n", duration.count());

	return 0;
}