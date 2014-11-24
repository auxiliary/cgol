#include <stdio.h>
#include <time.h>
#include <stdlib.h>>
#define N 256
__global__ void play(int *in, int *out)
{
    int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int live_cells = 0;
    if (bid * bdim + tid < 256)
    {
		// Check to see if the index is correct
		if (bid != 0 && tid != 0 && in[(bid - 1) * bdim + (tid - 1)])
				live_cells++; //Top left
		if (bid != 0 && in[(bid - 1) * bdim + tid])
				live_cells++; //Top	
		if (bid != 0 && tid != 63 && in[(bid - 1) * bdim + (tid + 1)])
				live_cells++; //Top right
		if (tid != 0 && in[(bid) * bdim + (tid - 1)])
				live_cells++; //left
		//Skipping itself
		if (tid != 63 && in[(bid) * bdim + (tid + 1)])
				live_cells++; //Right
		if (bid != 3 && tid != 0 && in[(bid + 1) * bdim + (tid - 1)])
				live_cells++; //Bottom left
		if (bid != 3 && tid != 63 && in[(bid + 1) * bdim + tid])
				live_cells++; //Bottom
		if (bid != 3 && tid != 63 && in[(bid + 1) * bdim + (tid + 1)])
				live_cells++; //Bottom right

		int is_live = in[bid * bdim + tid];
		out[bid * bdim + tid] = is_live;
		if ((is_live && live_cells < 2) || (is_live && live_cells > 3))
		{
			out[bid * bdim + tid] = 0;
		}
		else if (!is_live && live_cells == 3)
		{
			out[bid * bdim + tid] = 1;	
		}
    }
}

void print_board(int board[], int size)
{
    for (int i = 0;i < size; i++)
    {
		for (int j = 0; j < size; j++)
		{
			if (board[i * size + j] != 0 && board[i * size + j] != 1)
			{
				printf("*");	
			}
			else
				printf("%d ", board[i * size + j]);
		}
		printf("\n");
    }
	printf("\n\n");
}

int main(void) 
{
	srand(time(NULL));

	int size = 16;
	int no_blocks = 4;
	int no_threads = 64;
    int input[N], output[N];
    int *devin, *devout;

    cudaMalloc((void**)&devin, N*sizeof(int));
    cudaMalloc((void**)&devout, N*sizeof(int));

    for (int i = 0;i < size; i++)
    {
		for (int j = 0; j < size; j++)
		{
			input[i*size + j] = rand() % 2;
		}
    }
	print_board(input, size);

    cudaMemcpy(devin, input, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devout, output, size * size * sizeof(int), cudaMemcpyHostToDevice);

	// Call the kernel for one iteration
    play<<<no_blocks,no_threads>>>(devin, devout);

	// Copy back the output
    cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);

	print_board(output, size);

	// Free device memory
    cudaFree(devin);
    cudaFree(devout);

    return 0;
}
