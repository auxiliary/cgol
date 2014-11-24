#include <stdio.h>
#include <time.h>
#include <stdlib.h>>
#define N 16
__global__ void play(int *in, int *out)
{
    int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int live_cells = 0;
    if (bid * bdim + tid < 16)
    {
		// Check to see if the index is correct
		if (bid != 0 && tid != 0 && in[(bid - 1) * bdim + (tid - 1)])
				live_cells++; //Top left
		if (bid != 0 && in[(bid - 1) * bdim + tid])
				live_cells++; //Top	
		if (bid != 0 && tid != 3 && in[(bid - 1) * bdim + (tid + 1)])
				live_cells++; //Top right
		if (tid != 0 && in[(bid) * bdim + (tid - 1)])
				live_cells++; //left
		//Skipping itself
		if (tid != 3 && in[(bid) * bdim + (tid + 1)])
				live_cells++; //Right
		if (bid != 3 && tid != 0 && in[(bid + 1) * bdim + (tid - 1)])
				live_cells++; //Bottom left
		if (bid != 3 && in[(bid + 1) * bdim + tid])
				live_cells++; //Bottom
		if (bid != 3 && tid != 3 && in[(bid + 1) * bdim + (tid + 1)])
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
	__syncthreads();
}

void print_board(int board[], int size)
{
    for (int i = 0;i < size; i++)
    {
		for (int j = 0; j < size; j++)
		{
			if (board[i * size + j] != 0 && board[i * size + j] != 1)
			{
				printf("?");	
			}
			else
			{
				if (board[i * size + j])
				{
					printf("\u25A0");		
				}
				else
				{
					printf("\u25A1");
				}
			}
		}
		printf("\n");
    }
	printf("\n\n");
}

int main(void) 
{
	//srand(time(NULL));
	srand(2);
	int size = 4;
	int iterations = 100;
	int no_blocks = 4;
	int no_threads = 4;
    int input[N], output[N];
	//int input[16] = {1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0};
    int *devin, *devout, *devtemp;

    cudaMalloc((void**)&devin, N*sizeof(int));
    cudaMalloc((void**)&devout, N*sizeof(int));
	cudaMalloc((void**)&devtemp, N*sizeof(int));

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
	for (int i = 0;i<iterations;i++)
	{
		if (i == 0)
		{
			play<<<no_blocks,no_threads>>>(devin, devout);
		}
		else
		{
			play<<<no_blocks,no_threads>>>(devtemp, devout);
		}
		cudaMemcpy(devtemp, devout, size * size * sizeof(int), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);
		//print_board(output, size);
	}

	// Copy back the output
    cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);

	print_board(output, size);

	// Free device memory
    cudaFree(devin);
    cudaFree(devout);

    return 0;
}
