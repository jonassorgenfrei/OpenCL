// --------------------------------------------------------------------------------------
// Blocked Matrix Multipliplication kernel
// kernel: mat_mul 
// Purpose: compute the product of the multiplication of two matrices;
//          Using the well known blocked algorithm.  
//
//          To derive this algorithm, start with the naive
//          triply nested loop algorithm with a dot product 
//          for each element of C.  Decompose each loop 
//          into blocks of size blcksz.  This gives you 6
//          nested loops with three loops over blocks
//          and three loops over indices inside the blocks.
// 
//          Rearrange the loops to put the 3 loops over blocks 
//          at the outermost loops of the loop nest.  You'll
//          see that the three "inner" loops are just the 
//          regular matrix product between blocks.
//
//          The algorithms is simple.  Keeping all the indices
//          straight is not.  We will use the following 
//          conventions:
//
//             i,j,k            ... indices of full, global matrices 
//             Iblk, Jblk, Kblk ... indices of matrix blocks
//             iloc, jloc, kloc ... indices inside blocks
//
// input: A and B float matrices of dimension dim
// output: C float matrix of dimension dim holding the product of A * B
//

// It turns out that the compiler generates much better code if
// we "hardwire" this block size.  16 works well for an NVIDIA 
// GPU, 32 works well for a CPU
#define blksz 16

// __kernel declares a functions as a kernel (makes it visible to host code so it can be enqueued)
__kernel void mat_mul(
		const		int				N,
		__global	const		float* restrict A,		// __global address space qualifiers
		__global	const		float* restrict B,
		__global				float* restrict C,
		__local					float* restrict	Awrk,					// local shared by workitems in the work group
		__local					float* restrict	Bwrk)
{
	int kloc, Kblk;
	float Ctmp = 0.0f;

	//  This work-item will compute element C(i,j)
	const int i = get_global_id(0);
	const int j = get_global_id(1);

	//  Element C(i,j) is in block C(Iblk, JBlk)
	const int Iblk = get_group_id(0);
	const int Jblk = get_group_id(1);

	//  C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
	const int iloc = get_local_id(0);
	const int jloc = get_local_id(1);

	// the number of blocks are the same in each dimension
	const int Num_BLK = N / blksz;

	// setup the upper-left-corner (base address) for the A and
	// B block plus the increments to advance base addresses as
	// we loop over blocks
	int Abase = Jblk * N * blksz;
	const int Ainc = blksz;

	int Bbase = Iblk * blksz;
	const int Binc = blksz * N;

	// C(Iblk, Jblk) = (sum over Kblk) A(Iblk, Kblk)*B(Kblk, Jblk)
	for (Kblk = 0; Kblk < Num_BLK; Kblk++)
	{

		// load A(Iblk, Kblk) and B(Kblk, Jblk) into local memory.
		// Each work-item loads a single element of the two blocks
		// which are shared with the entire work-group

		Awrk[jloc * blksz + iloc] = A[Abase + jloc * N + iloc];
		Bwrk[jloc * blksz + iloc] = B[Bbase + jloc * N + iloc];

		barrier(CLK_LOCAL_MEM_FENCE);

		// compute dot products over local blocks to find thee
		// contribution to C(i,j) from this block
#pragma unroll
		for (kloc = 0; kloc < blksz; kloc++)
			Ctmp += Awrk[jloc * blksz + kloc] * Bwrk[kloc * blksz + iloc];

		barrier(CLK_LOCAL_MEM_FENCE);

		Abase += Ainc;
		Bbase += Binc;
	}

	// update global C matrix
	C[j * N + i] = Ctmp;
}