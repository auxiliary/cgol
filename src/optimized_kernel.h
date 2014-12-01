
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
