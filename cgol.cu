#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <unistd.h>
#include "optimized_kernel.h"
#include "simple_kernel.h"
#include "natural_indexed_kernel.h"

int SIZE, ITERATIONS, ANIMATE, BLOCKS, THREADS, SEED, UNOPTIMIZED, PRINT;
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
		if (c == 'u')
		{
			UNOPTIMIZED = 1;
		}
		if (c == 'p')
		{
			sscanf(argv[i++], "%d", &PRINT);
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
	int unoptimized_run = UNOPTIMIZED ? UNOPTIMIZED : 0;
	int print = PRINT != -1 ? PRINT : true;

	// Initialize random seed
	srand(SEED != -1 ? SEED : time(NULL));

	// Allocate space on host
	int *input = (int*)calloc(size * size, sizeof(int));
	int *output = (int*)calloc(size * size, sizeof(int));
    int *devin, *devout, *devtemp;

	// Allocate space on device
    cudaMalloc((void**)&devin, size * size * sizeof(int));
    cudaMalloc((void**)&devout, size * size * sizeof(int));
	cudaMalloc((void**)&devtemp, size * size * sizeof(int));

	// Generate random input
    for (int i = 0;i < size; i++)
    {
		for (int j = 0; j < size; j++)
		{
			input[i*size + j] = rand() % 2;
		}
    }

	if (print)
		print_board(input, size, 0);

	// Copy from host to device
    cudaMemcpy(devin, input, size * size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(devout, output, size * size * sizeof(int), cudaMemcpyHostToDevice);

	int shared_board_size = (no_threads + 2 * size) * sizeof(int);
	// Call the chosen kernel and time the run
	clock_t start = clock(), diff;
	if (unoptimized_run)
	{
		printf("Unoptimized run");
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
			if (animate == true)
			{
				system("clear");
				print_board(output, size, i);
				usleep(100000);
			}
		}
	}
	else
	{
		for (int i = 0;i<iterations;i++)
		{
			if (i == 0)
			{
				play_with_shared_memory<<<no_blocks,no_threads,shared_board_size>>>(devin, devout, size);
			}
			else
			{
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
	}

	// Copy back the output
    cudaMemcpy(output, devout, size * size * sizeof(int), cudaMemcpyDeviceToHost);
	
	if (print)
		print_board(output, size, iterations);

	// Calculate the time it took
	diff = clock() - start;
	int msec = diff * 1000 / CLOCKS_PER_SEC;
	printf("Time in kernel: %d seconds %d milliseconds\n", msec / 1000, msec % 1000);

	// Free device memory
    cudaFree(devin);
    cudaFree(devout);
	cudaFree(devtemp);

    return 0;
}

int main(int argc, char* argv[])
{
	SIZE = 0, ITERATIONS = 0, ANIMATE = -1, BLOCKS = 0, THREADS = 0, UNOPTIMIZED = 0, SEED = -1, PRINT = -1;
	arg_parse(argc, argv);
	run();
	return 0;
}

