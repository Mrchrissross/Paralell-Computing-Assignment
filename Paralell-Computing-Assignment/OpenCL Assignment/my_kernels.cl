//fixed 4 step reduce
__kernel void reduce_add_1(global const int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying

	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	if (((id % 2) == 0) && ((id + 1) < N))
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N))
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N))
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N))
		B[id] += B[id + 8];
}

//float conversion of the above function ^^ (fixed 4 step reduce)
__kernel void float_reduce_add_1(__global const float* A, __global float* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id]; //copy input to output

	barrier(CLK_GLOBAL_MEM_FENCE); //wait for all threads to finish copying

	//perform reduce on the output array
	//modulo operator is used to skip a set of values (e.g. 2 in the next line)
	//we also check if the added element is within bounds (i.e. < N)
	if (((id % 2) == 0) && ((id + 1) < N))
		B[id] += B[id + 1];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 4) == 0) && ((id + 2) < N))
		B[id] += B[id + 2];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 8) == 0) && ((id + 4) < N))
		B[id] += B[id + 4];

	barrier(CLK_GLOBAL_MEM_FENCE);

	if (((id % 16) == 0) && ((id + 8) < N))
		B[id] += B[id + 8];
}

//flexible step reduce 
__kernel void reduce_add_2(global const int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2) 
	{ //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N))
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//float conversion of the above function ^^ (flexible step reduce)
__kernel void float_reduce_add_2(__global const float* A, __global float* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	barrier(CLK_GLOBAL_MEM_FENCE);
	for (int i = 1; i < N; i *= 2)
	{ //i is a stride
		if (!(id % (i * 2)) && ((id + i) < N))
			B[id] += B[id + i];

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

//reduce using local memory (so called privatisation)
__kernel void reduce_add_3(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//float conversion of the above function ^^ (reduce using local memory)
__kernel void float_reduce_add_3(__global const float* A, __global float* B, local float* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//copy the cache to output array
	B[id] = scratch[lid];
}

//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
__kernel void reduce_add_4(global const int* A, global int* B, local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) 
		atomic_add(&B[0], scratch[lid]);
}

//float conversion of the above function ^^ (reduce using local memory + accumulation of local sums into a single location)
__kernel void float_reduce_add_4(__global const float *A, __global float *B, __local float *scratch)
{
	//		• work – a total number of operations
	//		• span – a total number of sequential steps

	// Get the global id of the work item (thread)
	int id = get_global_id(0);
	// Get the local id of the work item
	int lid = get_local_id(0);
	// Get the work group id
	int gid = get_group_id(0);
	// Get the amount of work items
	int N = get_local_size(0);

	// Cache all N values from global memory to local memory
	scratch[lid] = A[id];
	// The barriers between each step assure that the global memory operations are completed before the next step of the algorithm commences - Synchronisation!
	barrier(CLK_LOCAL_MEM_FENCE);

	// Loop through each work item, squaring the stride each time.
	for (int i = 1; i < N; i *= 2) 
	{
		// The modulo operator is used to choose the right index and how the stride changes
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] += scratch[lid + i];	// If the above statement is true, the two values are added together.

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// If the local id is 0...
	if (!lid) 
	{
		// Copy the cache to output array group
		B[gid] = scratch[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		// If the global id is 0...
		if (!id) 
		{
			// Loop through each of the groups and add elements to the output array
			for (int i = 1; i < get_num_groups(0); ++i)
				B[id] += B[i];
		}
	}
}

// conversion from atomic_min. Returns the lowest element.
__kernel void float_reduce_min(__global const float* A, __global float* B, __local float* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = (scratch[lid] < scratch[lid + i]) ? scratch[lid] : scratch[lid + i];
	//							^^
	//		if scratch[lid] is less than next element, scratch lid remains the same, else if it isn't then is become the next element.

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// If the local id is 0...
	if (!lid)
	{
		// Copy the cache to output array group
		B[gid] = scratch[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		// If the global id is 0...
		if (!id)
		{
			// Loop through each of the groups and add elements to the output array
			for (int i = 1; i < get_num_groups(0); ++i)
				B[id] = (B[i] < B[id]) ? B[i] : B[id];
	//							^^
	//		if B[i] is less than B[globalID] then B[i] remains unchanged, else it becomes B[globalID].
		}
	}
}

// Same function as above but returns the maximum element instead.
__kernel void float_reduce_max(__global const float* A, __global float* B, __local float* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int gid = get_group_id(0);
	int N = get_local_size(0);

	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = (scratch[lid] > scratch[lid + i]) ? scratch[lid] : scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if (!lid)
	{
		B[gid] = scratch[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		if (!id)
		{
			for (int i = 1; i < get_num_groups(0); ++i)
				B[id] = (B[i] > B[id]) ? B[i] : B[id];
		}
	}
}

// Parallel Selection Sort - More info found at: http://www.bealto.com/gpu-sorting_parallel-selection.html
// Selection Sort [Best/Worst: O(N^2)]
__kernel void float_selection_sort(__global const float *A, __global float *B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	float ikey = A[id];

	// Compute position of A[i] in output
	int pos = 0;

	for (int j = 0; j < N; j++)
	{
		float jkey = A[j]; // broadcasted

		bool smaller = (jkey < ikey) || (jkey == ikey && j < id); // in[j] < in[i] ?
		pos += (smaller) ? 1 : 0;
	}

	B[pos] = ikey;
}


// This sort is all around better than the one above.
// Both of these sorts were ran ten times and their averages were:

// float_selection_sort - mean time: 14193.7ms.
// float_selection_sort_local - mean time: 10402.7 ms

// Meaning that between the two, there was a difference of 3791 ms, or 3.8 seconds for each full sort of temperatures.

__kernel void float_selection_sort_local(__global const float *A, __global float *B, __local float *scratch)
{
	int id = get_global_id(0);	// current thread
	int N = get_global_size(0); // input size
	int LN = get_local_size(0); // workgroup size

	float ikey = A[id];			// input key for current thread

	// Compute position of A[i] in output
	int pos = 0;

	// Loop through the input (N) where the stride is the size of a workgroup (LN).
	for (int j = 0; j < N; j += LN)
	{
		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop through the workgroup from where the current local id is.
		for (int index = get_local_id(0); index < LN; index += LN)
			// Assign the position on the local array to that of the input array position j + index.
			scratch[index] = A[j + index];

		barrier(CLK_LOCAL_MEM_FENCE);

		// Loop through all local values
		for (int index = 0; index < LN; index++)
		{
			float jkey = scratch[index];  // assign jkey to the newly assigned local value assigned above. 

			// Determine whether the local value is smaller than the global poisiton on the input array.
			bool smaller = (jkey < ikey) || (jkey == ikey && (j + index) < id); // in[j] < in[i] ?
			// If it is smaller, then increase the pos by 1.
			pos += (smaller) ? 1 : 0;
		}
	}

	// Finally assign the current input value to the calculated position within the output array. 
	B[pos] = ikey;
}

void Swap(__global float* A, __global float* B)
{
	// Check if float A is larger than float B then swap
	if (*A > *B)
	{
		//		• Temp becomes A
		int temp = *A;

		//		• A becomes B
		*A = *B;

		//		• B becomes A
		*B = temp;
	}
}

// OddEven Sort [Best/Worst: O(N2^2)]
__kernel void sort_oddeven(__global float* A, __global float* B)
{
	int id = get_global_id(0);
	int	N = get_global_size(0);

	for (int i = 0; i < N; i += 2)
	{
		// Check if the id is even and next id is also less than the global size
		if ((id % 2 == 0) && (id + 1 < N))
			Swap(&A[id], &A[id + 1]);
						// Swap B[id] with B[id] + 1 if the element is lower than the other

		barrier(CLK_GLOBAL_MEM_FENCE);

		// Same as the above but odd.
		if ((id % 2 == 1) && (id + 1 < N))
			Swap(&A[id], &A[id + 1]);

		barrier(CLK_GLOBAL_MEM_FENCE);

		B[id] = A[id];
		B[id + 1] = A[id + 1];
	}
}

__kernel void float_reduce_variance(__global const float* input, __global float* output, float mean)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	
	// • The variance is obtained by subtracting each element by the mean, and then squaring the result.
	if (id < N)
	{									
		output[id] = input[id] - mean;
		output[id] = (output[id] * output[id]);
	}
}

//a very simple histogram implementation
__kernel void hist_simple(global const int* A, global int* H)
{
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

//Hillis-Steele basic inclusive scan
//requires additional buffer B to avoid data overwrite 
__kernel void scan_hs(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	global int* C;

	for (int stride = 1; stride < N; stride *= 2) 
	{
		B[id] = A[id];

		if (id >= stride)
			B[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

		C = A; A = B; B = C; //swap A & B between steps
	}
}

//a double-buffered version of the Hillis-Steele inclusive scan
//requires two additional input arguments which correspond to two local buffers
__kernel void scan_add(__global const int* A, global int* B, local int* scratch_1, local int* scratch_2)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);
	local int *scratch_3;//used for buffer swap

	//cache all N values from global memory to local memory
	scratch_1[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) 
	{
		if (lid >= i)
			scratch_2[lid] = scratch_1[lid] + scratch_1[lid - i];
		else
			scratch_2[lid] = scratch_1[lid];

		barrier(CLK_LOCAL_MEM_FENCE);

		//buffer swap
		scratch_3 = scratch_2;
		scratch_2 = scratch_1;
		scratch_1 = scratch_3;
	}

	//copy the cache to output array
	B[id] = scratch_1[lid];
}

//Blelloch basic exclusive scan
__kernel void scan_bl(global int* A)
{
	int id = get_global_id(0);
	int N = get_global_size(0);
	int t;

	//up-sweep
	for (int stride = 1; stride < N; stride *= 2) 
	{
		if (((id + 1) % (stride * 2)) == 0)
			A[id] += A[id - stride];

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}

	//down-sweep
	if (id == 0)
		A[N - 1] = 0;//exclusive scan

	barrier(CLK_GLOBAL_MEM_FENCE); //sync the step

	for (int stride = N / 2; stride > 0; stride /= 2) 
	{
		if (((id + 1) % (stride * 2)) == 0) 
		{
			t = A[id];
			A[id] += A[id - stride]; //reduce 
			A[id - stride] = t;		 //move
		}

		barrier(CLK_GLOBAL_MEM_FENCE); //sync the step
	}
}

//calculates the block sums
__kernel void block_sum(global const int* A, global int* B, int local_size)
{
	int id = get_global_id(0);
	B[id] = A[(id + 1)*local_size - 1];
}

//simple exclusive serial scan based on atomic operations - sufficient for small number of elements
__kernel void scan_add_atomic(global int* A, global int* B)
{
	int id = get_global_id(0);
	int N = get_global_size(0);

	for (int i = id + 1; i < N; i++)
		atomic_add(&B[i], A[id]);
}

//adjust the values stored in partial scans by adding block sums to corresponding blocks
__kernel void scan_add_adjust(global int* A, global const int* B)
{
	int id = get_global_id(0);
	int gid = get_group_id(0);
	A[id] += B[gid];
}

