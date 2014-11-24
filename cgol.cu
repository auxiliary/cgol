#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

__global__ void play_with_row_based_index(int *in, int *out, int size)
{
    int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int live_cells = 0;
	int max = size * size;
	int my_id = bid * bdim + tid;
	int mod = my_id % size;
	// Check to see if the index is correct
	if (mod != 0 && my_id + size < max && in[my_id + size - 1])						// Top left
	{
		live_cells++;
	}
	if (my_id + size < max && in[my_id + size])										// Top
	{
		live_cells++;
	}
	if (mod != size - 1 && my_id + size < max && in[my_id + size + 1])				// Top right
	{
		live_cells++;	
	}
	if (mod != 0 && in[my_id - 1])													// Left
	{
		live_cells++;	
	}
	if (mod != size - 1 && in[my_id + 1])											// Right
	{
		live_cells++;	
	}
	if (my_id - size>= 0 && mod != 0 && in[my_id - size - 1])						// Bottom left
	{
		live_cells++;	
	}
	if (my_id - size >= 0 && in[my_id - size])										// Bottom
	{
		live_cells++;	
	}
	if (my_id - size >= 0 && mod != size - 1 && in[my_id - size + 1])				// Bottom right
	{
		live_cells++;	
	}

	int is_live = in[my_id];
	out[my_id] = is_live;
	if ((is_live && live_cells < 2) || (is_live && live_cells > 3))
	{
		out[my_id] = 0;
	}
	else if (!is_live && live_cells == 3)
	{
		out[my_id] = 1;	
	}
	__syncthreads();
}

__global__ void play(int *in, int *out)
{
    int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int gdim = gridDim.x;
	int live_cells = 0;
    if (bid * bdim + tid < bdim * gdim)
    {
		// Check to see if the index is correct
		if (bid != 0 && tid != 0 && in[(bid - 1) * bdim + (tid - 1)])
				live_cells++; //Top left
		if (bid != 0 && in[(bid - 1) * bdim + tid])
				live_cells++; //Top	
		if (bid != 0 && tid != bdim - 1 && in[(bid - 1) * bdim + (tid + 1)])
				live_cells++; //Top right
		if (tid != 0 && in[(bid) * bdim + (tid - 1)])
				live_cells++; //left
		//Skipping itself
		if (tid != bdim - 1 && in[(bid) * bdim + (tid + 1)])
				live_cells++; //Right
		if (bid != gdim - 1 && tid != 0 && in[(bid + 1) * bdim + (tid - 1)])
				live_cells++; //Bottom left
		if (bid != gdim - 1 && in[(bid + 1) * bdim + tid])
				live_cells++; //Bottom
		if (bid != gdim - 1 && tid != bdim - 1 && in[(bid + 1) * bdim + (tid + 1)])
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

void print_board(int board[], int size, int iteration)
{
	printf("Iteration %d\n", iteration);
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
					printf("\u25A3 ");		
				}
				else
				{
					printf("\u25A2 ");
				}
			}
		}
		printf("\n");
    }
	printf("\n\n");
}

int main(void) 
{
	/*
	while ((opt = getopt(argc, argv, "ilw")) != -1)
	{
		switch (opt)	
		{
				
		}
	}
	*/
	//srand(time(NULL));
	srand(6);
	int size = 32;
	int iterations = 20;
	int no_blocks = 8;
	int no_threads = 128;
	int *input = (int*)calloc(size * size, sizeof(int));
	int *output = (int*)calloc(size * size, sizeof(int));
	//int input[16] = {1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0};
    int *devin, *devout, *devtemp;

    cudaMalloc((void**)&devin, size * size * sizeof(int));
    cudaMalloc((void**)&devout, size * size * sizeof(int));
	cudaMalloc((void**)&devtemp, size * size * sizeof(int));

    for (int i = 0;i < size; i++)
    {
		for (int j = 0; j < size; j++)
		{
			input[i*size + j] = rand() % 2;
		}
    }
	print_board(input, size, 0);

    cudaMemcpy(devin, input, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devout, output, size * size * sizeof(int), cudaMemcpyHostToDevice);

	// Call the kernel for one iteration
	for (int i = 0;i<iterations;i++)
	{
		if (i == 0)
		{
			play_with_row_based_index<<<no_blocks,no_threads>>>(devin, devout, size);
		}
		else
		{
			play_with_row_based_index<<<no_blocks,no_threads>>>(devtemp, devout, size);
		}
		cudaMemcpy(devtemp, devout, size * size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);
		//system("clear");
		print_board(output, size, i);
		usleep(100000);
	}

	// Copy back the output
    cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);

	print_board(output, size, iterations);

	// Free device memory
    cudaFree(devin);
    cudaFree(devout);
	cudaFree(devtemp);

    return 0;
}
