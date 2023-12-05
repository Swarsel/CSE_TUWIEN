#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>

__global__ void XRP(double *solution, double *p, double* r, double* Ap, double* result,
                            int N, double alpha, double beta) {

  double dot{0};
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x){
    solution[i] += alpha * p[i];
    r[i] -= alpha * Ap[i];
    p[i] = r[i] + beta * p[i];
    dot += r[i] * r[i];
  }
  for (int j=warpSize/2; j>0; j=j/2) {
    dot += __shfl_xor_sync(0xffffffff, dot, j);
  }

  if (threadIdx.x % warpSize == 0) {
    atomicAdd(result, dot);
  }

}

__global__ void API(int N, int *csr_rowoffsets, int *csr_colindices, double *csr_values,
                           double *p, double *Ap, double *result_pAp, double *result_ApAp) {

  double pAp{0}, ApAp{0};
  for (int row = blockIdx.x * blockDim.x + threadIdx.x; row < N; row += blockDim.x * gridDim.x) {
    double sum = 0;
    for (int jj = csr_rowoffsets[row]; jj < csr_rowoffsets[row + 1]; ++jj) {
      sum += csr_values[jj] * p[csr_colindices[jj]];
    }
    Ap[row] = sum;
    pAp += p[row] * Ap[row];
    ApAp += Ap[row] * Ap[row];
  }

  for (int j=warpSize/2; j>0; j=j/2) {
    pAp += __shfl_xor_sync(0xffffffff, pAp, j);
    ApAp += __shfl_xor_sync(0xffffffff, ApAp, j);
  }

  if (threadIdx.x % warpSize == 0) {
    atomicAdd(result_pAp, pAp);
    atomicAdd(result_ApAp, ApAp);
  }
}


// y = A * x
__global__ void cuda_csr_matvec_product(int N, int *csr_rowoffsets,
                                        int *csr_colindices, double *csr_values,
                                        double *x, double *y)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    double sum = 0;
    for (int k = csr_rowoffsets[i]; k < csr_rowoffsets[i + 1]; k++) {
      sum += csr_values[k] * x[csr_colindices[k]];
    }
    y[i] = sum;
  }
}

// x <- x + alpha * y
__global__ void cuda_vecadd(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] += alpha * y[i];
}

// x <- y + alpha * x
__global__ void cuda_vecadd2(int N, double *x, double *y, double alpha)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x)
    x[i] = y[i] + alpha * x[i];
}

// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result)
{
  __shared__ double shared_mem[512];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0) atomicAdd(result, shared_mem[0]);
}



/** Implementation of the conjugate gradient algorithm.
 *
 *  The control flow is handled by the CPU.
 *  Only the individual operations (vector updates, dot products, sparse
 * matrix-vector product) are transferred to CUDA kernels.
 *
 *  The temporary arrays p, r, and Ap need to be allocated on the GPU for use
 * with CUDA. Modify as you see fit.
 */
void conjugate_gradient(int N, // number of unknows
                        int *csr_rowoffsets, int *csr_colindices,
                        double *csr_values, double *rhs, double *solution)
//, double *init_guess)   // feel free to add a nonzero initial guess as needed
{
  // initialize timer
  Timer timer;

  // clear solution vector (it may contain garbage values):
  std::fill(solution, solution + N, 0);

  // initialize work vectors:
  double alpha, beta;
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_scalar, *cuda_scalar2;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_scalar, sizeof(double));
  cudaMalloc(&cuda_scalar2, sizeof(double));

  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

  const double zero = 0;
  double residual_norm_squared = 0;
  cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_r, cuda_r, cuda_scalar);
  cudaMemcpy(&residual_norm_squared, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);

  double initial_residual_squared = residual_norm_squared;

  cuda_csr_matvec_product<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);

  cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_scalar2, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_p, cuda_Ap, cuda_scalar);
  cuda_dot_product<<<512, 512>>>(N, cuda_Ap, cuda_Ap, cuda_scalar2);
  cudaMemcpy(&alpha, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(&beta, cuda_scalar2, sizeof(double), cudaMemcpyDeviceToHost);
  alpha = residual_norm_squared / alpha;
  beta = alpha*alpha*beta / residual_norm_squared - 1;

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {

    cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(cuda_scalar2, &zero, sizeof(double), cudaMemcpyHostToDevice);

    XRP<<<512, 512>>>(cuda_solution, cuda_p, cuda_r, cuda_Ap, cuda_scalar, N, alpha, beta);
    cudaMemcpy(&residual_norm_squared, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
    API<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap, cuda_scalar, cuda_scalar2);

    cudaMemcpy(&alpha, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&beta, cuda_scalar2, sizeof(double), cudaMemcpyDeviceToHost);

    alpha = residual_norm_squared / alpha;
    beta = (alpha*alpha*beta / residual_norm_squared) - 1;

    if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
      break;
    }

    if (iters > 10000)
      break; // solver didn't converge
    ++iters;

  }
  cudaMemcpy(solution, cuda_solution, sizeof(double) * N, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();
  // std::cout << "Time elapsed: " << timer.get() << " (" << timer.get() / iters << " per iteration)" << std::endl;

  // if (iters > 10000)
  //   std::cout << "Conjugate Gradient did NOT converge within 10000 iterations"
  //             << std::endl;
  // else
  //   std::cout << "Conjugate Gradient converged in " << iters << " iterations."
  //             << std::endl;

  cudaFree(cuda_p);
  cudaFree(cuda_r);
  cudaFree(cuda_Ap);
  cudaFree(cuda_solution);
  cudaFree(cuda_scalar);
  cudaFree(cuda_scalar2);
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int N1, int M1) {

  int N = N1 *
          M1; // number of unknows to solve for

  // std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix
  // a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros
  //       per row in the system matrix, so we can allocate accordingly.
  //
  int nval = (5 * (N1-2) * (M1-2)) + (3 * 4)  + (4 * 2 * (N1 - 2)) + (4 * 2 * (M1-2));
  int *csr_rowoffsets = (int *)malloc(sizeof(int) * (N + 1));
  int *csr_colindices = (int *)malloc(sizeof(int) * nval);
  double *csr_values = (double *)malloc(sizeof(double) * nval);

  int *cuda_csr_rowoffsets, *cuda_csr_colindices;
  double *cuda_csr_values;
  //
  // fill CSR matrix with values
  //
  generate_fdm_laplace(N1, csr_rowoffsets, csr_colindices,
                       csr_values);

  //
  // Allocate solution vector and right hand side:
  //
  double *solution = (double *)malloc(sizeof(double) * N);
  double *rhs = (double *)malloc(sizeof(double) * N);
  std::fill(rhs, rhs + N, 1);

  //
  // Allocate CUDA-arrays //
  //
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(int) * (N + 1));
  cudaMalloc(&cuda_csr_colindices, sizeof(int) * nval);
  cudaMalloc(&cuda_csr_values, sizeof(double) * nval);
  cudaMemcpy(cuda_csr_rowoffsets, csr_rowoffsets, sizeof(int) * (N + 1), cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_colindices, csr_colindices, sizeof(int) * nval,   cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_csr_values,     csr_values,     sizeof(double) * nval,   cudaMemcpyHostToDevice);

  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  cudaDeviceSynchronize();
  Timer timer;
  double time{0};
  timer.reset();
  conjugate_gradient(N, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);
  cudaDeviceSynchronize();
  time = timer.get();

  std::cout << time;

  //
  // Check for convergence:
  //
  // double residual_norm = relative_residual(N, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  // std::cout << "Relative residual norm: " << residual_norm
  //           << " (should be smaller than 1e-6)" << std::endl;

  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
  free(solution);
  free(rhs);
  free(csr_rowoffsets);
  free(csr_colindices);
  free(csr_values);
}


int main(int argc, char *argv[]) {
  int N = std::atoi(argv[1]);
  int M = std::atoi(argv[2]);
  solve_system(N, M); // solves a system with 100*100 unknowns
  return EXIT_SUCCESS;
}
