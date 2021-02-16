// --------------------------------------------------------------------------------------
// kernel: mat_mul 
// Purpose: compute the product of the multiplication of two matrices;
//          computing a dot product for each element of the product matrix
// input: A and B float matrices of dimension dim
// output: C float matrix of dimension dim holding the product of A * B
//

// __kernel declares a functions as a kernel (makes it visible to host code so it can be enqueued)
__kernel void mat_mul(
	const int N,
	__global float* A,		// __global address space qualifiers
	__global float* B,
	__global float* C
	)
{
	int i, j, k;
	
	// work-item_co-ordinates
	i = get_global_id(0);
	j = get_global_id(1);
	
	// use local scalar for intermediate C element values
	float tmp = 0.0f;

	if ((i < N) && (j < N))
	{
		for (k = 0; k < N; k++) {
			// C(i,j) = sum(over k) A(i,k)*B(k,j)
			tmp += A[i * N + k] * B[k * N + j];
		}

		// write result to C
		C[i * N + j] = tmp;
	}
}