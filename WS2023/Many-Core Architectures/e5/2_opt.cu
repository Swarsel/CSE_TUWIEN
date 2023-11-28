
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include "poisson2d.hpp"
#include "timer.hpp"

/** Computes y = A*x for a sparse matrix A in CSR format and vector x,y  */
__global__
void csr_matvec_product(int N, int *rowoffsets, int *colindices, double *values, double *x, double *y) {
    for (int row = blockDim.x * blockIdx.x + threadIdx.x; row < N; row += gridDim.x * blockDim.x) {
        double val = 0;
        for (int jj = rowoffsets[row]; jj < rowoffsets[row+1]; ++jj) {
            val += values[jj] * x[colindices[jj]];
        }
        y[row] = val;
    }
}


__global__ void dot(int N, double *x, double *y, double *results) {
    double alpha1{0};
    unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_id == 0) *results = 0;
    for(int j = thread_id; j < N; j += blockDim.x*gridDim.x) {
        alpha1 += x[j] * y[j];
    }

    for (int j=warpSize/2; j>0; j=j/2) {
        alpha1 += __shfl_xor_sync(0xffffffff, alpha1, j);
    }

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(results, alpha1);
    }
}

__global__
void xrIterate(int N, double *x, double *p, double *r, double *Ap, double mod) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
            x[i] += mod * p[i];
            r[i] -= mod * Ap[i];
        }
}

__global__
void vecIterate(int N, double *out, double *in1, double *in2, double mod) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
            out[i] = in1[i] + mod * in2[i];
        }
}

/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use with CUDA.
 *  Modify as you see fit.
 */
void conjugate_gradient(size_t N,  // number of unknows
                        int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                        double *rhs,
                        double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
    // clear solution vector (it may contain garbage values):
    std::fill(solution, solution + N, 0);

    // initialize work vectors:
    double *p = (double*)malloc(sizeof(double) * N);
    double *r = (double*)malloc(sizeof(double) * N);
    double *Ap = (double*)malloc(sizeof(double) * N);

    // CPU variables
    double rr{0};
    double alpha{0};
    double beta{0};

    // line 2: initialize r and p:
    std::copy(rhs, rhs+N, p);
    std::copy(rhs, rhs+N, r);

    // initialize variables for GPU
    int *cuda_csr_rowoffsets, *cuda_csr_colindices;
    double *cuda_csr_values, *cuda_p, *cuda_r, *cuda_Ap, *cuda_out, *cuda_solution;
    cudaMalloc(&cuda_csr_rowoffsets, sizeof(int) * (N + 1));
    cudaMalloc(&cuda_csr_colindices, sizeof(int) * 5 * N);
    cudaMalloc(&cuda_csr_values, sizeof(double) * 5 * N);
    cudaMalloc(&cuda_p, sizeof(double) * N);
    cudaMalloc(&cuda_r, sizeof(double) * N);
    cudaMalloc(&cuda_Ap, sizeof(double) * N);
    cudaMalloc(&cuda_out, sizeof(double) * 1);
    cudaMalloc(&cuda_solution, sizeof(double) * N);
    // cudaMalloc(&cuda_rprp, sizeof(double) * 1);

    cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(int) * 5 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_csr_values, csr_values, sizeof(double) * 5 * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_p, p, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_r, r, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);


    //    cudaMemset(&cuda_rr, 0, 1);
    dot<<<256,256>>>(N, cuda_r, cuda_r, cuda_out);
    // cudaDeviceSynchronize();
    cudaMemcpy(&rr, cuda_out, sizeof(double), cudaMemcpyDeviceToHost);
    // std::cout << rr << std::endl;
    double rr0 = rr;

    int iters = 0;
    while (1) {

        // line 4: A*p:
        csr_matvec_product<<<128, 128>>>(N, cuda_csr_rowoffsets, cuda_csr_colindices,
                                                   cuda_csr_values, cuda_p, cuda_Ap);

        dot<<<128, 128>>>(N, cuda_p, cuda_Ap, cuda_out);
        cudaMemcpy(&alpha, cuda_out, sizeof(double), cudaMemcpyDeviceToHost);
        alpha = rr / alpha;

        xrIterate<<<128,128>>>(N, cuda_solution, cuda_p, cuda_r, cuda_Ap, alpha);

        beta = rr;

        dot<<<128, 128>>>(N, cuda_r, cuda_r, cuda_out);
        cudaMemcpy(&rr, cuda_out, sizeof(double), cudaMemcpyDeviceToHost);

        if (std::sqrt(rr/rr0) < 1e-6) break;

        beta = rr / beta;

        vecIterate<<<128, 128>>>(N, cuda_p, cuda_r, cuda_p, beta);
        cudaDeviceSynchronize();

        if (iters > 10000) break;  // solver didn't converge
        ++iters;
    }

    cudaMemcpy(solution, cuda_solution, N * sizeof(double), cudaMemcpyDeviceToHost);


    free(p);
    free(r);
    free(Ap);
    cudaFree(cuda_csr_rowoffsets);
    cudaFree(cuda_csr_colindices);
    cudaFree(cuda_csr_values);
    cudaFree(cuda_p);
    cudaFree(cuda_r);
    cudaFree(cuda_Ap);
    cudaFree(cuda_solution);
    cudaFree(cuda_out);

}



/** Solve a system with `points_per_direction * points_per_direction` unknowns */
void solve_system(size_t points_per_direction) {

    size_t N = points_per_direction * points_per_direction; // number of unknows to solve for

    //std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

    //
    // Allocate CSR arrays.
    //
    // Note: Usually one does not know the number of nonzeros in the system matrix a-priori.
    //       For this exercise, however, we know that there are at most 5 nonzeros per row in the system matrix, so we can allocate accordingly.
    //
    int *csr_rowoffsets =    (int*)malloc(sizeof(double) * (N+1));
    int *csr_colindices =    (int*)malloc(sizeof(double) * 5 * N);
    double *csr_values  = (double*)malloc(sizeof(double) * 5 * N);

    //
    // fill CSR matrix with values
    //
    generate_fdm_laplace(points_per_direction, csr_rowoffsets, csr_colindices, csr_values);

    //
    // Allocate solution vector and right hand side:
    //
    double *solution = (double*)malloc(sizeof(double) * N);
    double *rhs      = (double*)malloc(sizeof(double) * N);
    std::fill(rhs, rhs + N, 1);

    //
    // Call Conjugate Gradient implementation (CPU arrays passed here; modify to use GPU arrays)
    //
    Timer timer;
    double time;
    timer.reset();
    conjugate_gradient(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
    time = timer.get();
    std::cout << time;

    //
    // Check for convergence:
    //
    double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
    //std::cout << "Relative residual norm: " << residual_norm << " (should be smaller than 1e-6)" << std::endl;

    free(solution);
    free(rhs);
    free(csr_rowoffsets);
    free(csr_colindices);
    free(csr_values);

}


int main(int argc, char *argv[]) {

    solve_system(std::atoi(argv[1])); // solves a system with 100*100 unknowns
    return EXIT_SUCCESS;
}
