#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>

int SIZE, ITERATIONS, ANIMATE, BLOCKS, THREADS, SEED, SHARED_MEMORY;
__global__ void play_with_shared_memory(int *in, int *out, int size)
{
    int bid = blockIdx.x;
	int tid = threadIdx.x;
	int bdim = blockDim.x;
	int live_cells = 0;
	int max = size * size;
	int my_id = bid * bdim + tid;
	int mod = my_id % size;
	extern __shared__ int local_board[];

	local_board[tid + size] = in[my_id];
	// Grab neighbors from next block if possible
	if (my_id % bdim >= bdim - size && my_id + size < max)
	{
		local_board[tid + 2 * size] = in[my_id + size];
	}
	// Grab neighbors from previous block if possible
	if (my_id % bdim < size && my_id - size >= 0)
	{
		local_board[tid] = in[my_id - size];
	}
	// Local Id
	int lid = tid + size;
	__syncthreads();

	// Check to see if the index is correct
	if (mod != 0 && my_id + size < max && local_board[lid + size - 1])						// Top left
	{
		live_cells++;
	}
	if (my_id + size < max && local_board[lid + size])													// Top
	{
		live_cells++;
	}
	if (mod != size - 1 && my_id + size < max && local_board[lid + size + 1])				// Top right
	{
		live_cells++;	
	}
	if (mod != 0 && local_board[lid - 1])													// Left
	{
		live_cells++;	
	}
	if (mod != size - 1 && local_board[lid + 1])											// Right
	{
		live_cells++;	
	}
	if (my_id - size>= 0 && mod != 0 && local_board[lid - size - 1])						// Bottom left
	{
		live_cells++;	
	}
	if (my_id - size >= 0 && local_board[lid - size])										// Bottom
	{
		live_cells++;	
	}
	if (my_id - size >= 0 && mod != size - 1 && local_board[lid - size + 1])				// Bottom right
	{
		live_cells++;	
	}

	int is_live = local_board[lid];
	int result = is_live;
	if ((is_live && live_cells < 2) || (is_live && live_cells > 3))
	{
		result = 0;
	}
	else if (!is_live && live_cells == 3)
	{
		result = 1;	
	}
	out[my_id] = result;

	__syncthreads();
}

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
	if (iteration != -1)
	{
		printf("Iteration %d\n", iteration);
	}
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

void arg_parse(int argc, char *argv[])
{
	int i = 1;
	char c;
	while(i < argc)
	{
		sscanf(argv[i++], "%c", &c);
		if (c == 's')
		{
			sscanf(argv[i++], "%d", &SIZE);
		}
		if (c == 'a')
		{
			ANIMATE = 1;	
			printf("fu");
		}
		if (c == 'i')
		{
			sscanf(argv[i++], "%d", &ITERATIONS);
		}
		if (c == 'b')
		{
			sscanf(argv[i++], "%d", &BLOCKS);
		}
		if (c == 't')
		{
			sscanf(argv[i++], "%d", &THREADS);
		}
		if (c == 'e')
		{
			sscanf(argv[i++], "%d", &SEED);
		}
		if (c == 'h')
		{
			SHARED_MEMORY = 1;
		}
	}
}

int run()
{
	int animate = ANIMATE != -1 ? ANIMATE : false;
	int size = SIZE ? SIZE : 32;
	int iterations = ITERATIONS ? ITERATIONS : 30;
	int no_blocks = BLOCKS ? BLOCKS : size;
	int no_threads = THREADS ? THREADS : size;
	srand(SEED != -1 ? SEED : time(NULL));
	int *input = (int*)calloc(size * size, sizeof(int));
	int *output = (int*)calloc(size * size, sizeof(int));
	/*int input[16] = {	0, 0, 0, 0, 
						1, 1, 1, 1, 
						0, 0, 0, 0, 
						0, 0, 0, 0};*/
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

	int shared_board_size = (no_threads + 2 * size) * sizeof(int);
	// Call the kernel for one iteration
	clock_t start = clock(), diff;
	for (int i = 0;i<iterations;i++)
	{
		if (i == 0)
		{
			//play_with_row_based_index<<<no_blocks,no_threads,shared_board_size>>>(devin, devout, size);
			play_with_shared_memory<<<no_blocks,no_threads,shared_board_size>>>(devin, devout, size);
		}
		else
		{
			//play_with_row_based_index<<<no_blocks,no_threads,shared_board_size>>>(devtemp, devout, size);
			play_with_shared_memory<<<no_blocks,no_threads,shared_board_size>>>(devtemp, devout, size);
		}
		cudaMemcpy(devtemp, devout, size * size * sizeof(int), cudaMemcpyDeviceToDevice);
		cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);
		if (animate == true)
		{
			system("clear");
			print_board(output, size, i);
			usleep(100000);
		}
	}

	// Copy back the output
    cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);
	
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Time in kernel: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	print_board(output, size, iterations);

	// Free device memory
    cudaFree(devin);
    cudaFree(devout);
	cudaFree(devtemp);

    return 0;
}

int main(int argc, char* argv[])
{
	SIZE = 0, ITERATIONS = 0, ANIMATE = -1, BLOCKS = 0, THREADS = 0, SHARED_MEMORY = 0, SEED = -1;
	arg_parse(argc, argv);
	run();
	return 0;
}
