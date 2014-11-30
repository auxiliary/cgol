
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
