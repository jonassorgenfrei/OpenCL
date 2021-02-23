//------------------------------------------------------------------------------
//
//  Matrix library for OpenCL Tests
//
//------------------------------------------------------------------------------

#ifndef __MATRIX_LIB_HDR
#define __MATRIX_LIB_HDR

#include <cstdio>
#include <vector>


/// <summary>
/// Multiplies the Matrix A with the Matrix B each of dimension N and writes the result to C.
/// Serial CPU implementation
/// </summary>
/// <param name="dim">The dimension of the matrices</param>
/// <param name="A">Matrix A</param>
/// <param name="B">Matrix B</param>
/// <param name="C">Matrix C</param>
void mat_mul(int dim, std::vector<float>& A, std::vector<float>& B, std::vector<float>& C)
{
    int i, j, k;
    float tmp;

    for (i = 0; i < dim; i++) {
        for (j = 0; j < dim; j++) {
            tmp = 0.0f;
            for (k = 0; k < dim; k++) {
                // C(i,j) = sum(over k) A(i,k)*B(k,j)
                tmp += A[i * dim + k] * B[k * dim + j];
            }
            C[i * dim + j] = tmp;
        }
    }
}

/// <summary>
/// Function to initialize the input matrix
/// </summary>
/// <param name="N">The N dimension of the matrix</param>
/// <param name="M">The M dimension of the matrix</param>
/// <param name="mat">The matrix</param>
/// <param name="value">The value to put in each field of the matrix</param>
void initmat(int N, int M, std::vector<float>& mat, float value) {
	/* Initialize matrices */

	for (int i = 0; i < N; i++)
		for (int j = 0; j < M; j++)
			mat[i * N + j] = value;

};


/// <summary>
/// Function to fill Btrans(N,N) with transpose of B(N,N)
/// </summary>
/// <param name="N">The dimension</param>
/// <param name="B">The matrix B</param>
/// <param name="Btrans">The transpose of B</param>
void trans(int N, std::vector<float>& B, std::vector<float>& Btrans)
{
    int i, j;

    for (i = 0; i < N; i++)
        for (j = 0; j < N; j++)
            Btrans[j * N + i] = B[i * N + j];
}



#endif