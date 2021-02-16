// --------------------------------------------------------------------------------------
// kernel: mat_mul 
// Purpose: compute the product of the multiplication of two matrices;
//          computing a dot product for each element of the product matrix
//          it computes one work item per row of C
// input: A and B float matrices of dimension dim
// output: C float matrix of dimension dim holding the product of A * B
//

// __kernel declares a functions as a kernel (makes it visible to host code so it can be enqueued)
__kernel void mat_mul(
	const int N,
	__global float* A,		// __global address space qualifiers
	__global float* B,
	__global float* C,
	__local float* Bwrk)	// local shared by workitems in the work group
{
	int j, k;

	// work-item_co-ordinates
	int i = get_global_id(0);

	int iloc = get_local_id(0);
	int nloc = get_local_size(0);

	// local memory initialization ORDER as CONST size of array
	float Awrk[1024];

	float tmp;

	if (i < N) {
		// copy row of A into private memory
		for (k = 0; k < N; k++)
			Awrk[k] = A[i * N + k];


		for (j = 0; j < N; j++) {

			// pass a work array in local memoy to hold a column 
			// of B
			// all the work-items do the copy 
			// "in parallel" using a cyclic loop distribution
			// (hence why we need iloc and nloc)

			for (k = iloc; k < N; k += nloc) {
				Bwrk[k] = B[k * N + k];
			}

			barrier(CLK_LOCAL_MEM_FENCE);

			// use local scalar for intermediate C element values
			tmp = 0.0f;
			for (k = 0; k < N; k++) {
				// C(i,j) = sum(over k) A(i,k)*B(k,j)
				tmp += Awrk[k] * Bwrk[k];
			}

			// write result to C
			C[i * N + j] = tmp;

			barrier(CLK_LOCAL_MEM_FENCE);
		}
	}
}