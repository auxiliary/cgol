
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
