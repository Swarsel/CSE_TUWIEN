#include "poisson2d.hpp"
#include "timer.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

__global__ void count_nnz(int *row_offsets, int N, int M) {
  for (int row = blockDim.x * blockIdx.x + threadIdx.x;
       row < N*M;
       row += gridDim.x * blockDim.x)
    {
      int nnz_for_this_node = 1;
      int i = row / N;
      int j = row % N;
      if (i > 0) nnz_for_this_node += 1;
      if (j > 0) nnz_for_this_node += 1;
      if (i < N-1) nnz_for_this_node += 1;
      if (j < M-1) nnz_for_this_node += 1;
      row_offsets[row] = nnz_for_this_node;
    }
}

__global__ void assembleA(int *row_offsets, int *col_indices, double *values, int N, int M) {
  for (int row = blockDim.x * blockIdx.x + threadIdx.x;
       row < N*M;
       row += gridDim.x * blockDim.x) {
    int i = row / N;
    int j = row % N;
    int this_row_offset = row_offsets[row];
    // diagonal entry
    col_indices[this_row_offset] = i * N + j;
    values[this_row_offset] = 4;
    this_row_offset += 1;
    if (i > 0) { // bottom neighbor
      col_indices[this_row_offset] = (i-1) * N + j;
      values[this_row_offset] = -1;
      this_row_offset += 1;
    }
    if (j > 0) { /* similarly */
      col_indices[this_row_offset] = i + N * (j-1);
      values[this_row_offset] = -1;
      this_row_offset += 1;
    }
    if (i < N-1) { /* similarly */
      col_indices[this_row_offset] = (i+1) * N + j;
      values[this_row_offset] = -1;
      this_row_offset += 1;
    }
    if (j < M-1) { /* similarly */
      col_indices[this_row_offset] = i + N * (j+1);
      values[this_row_offset] = -1;
      this_row_offset += 1;
    }
  }
}
__global__ void scan_kernel_1(int const *X,
                              int *Y,
                              int N,
                              int *carries)
{
  __shared__ int shared_buffer[256];
  int my_value;

  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;

  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < N) ? X[i] : 0;

    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();

    // exclusive scan requires us to write a zero value at the beginning of each block
    my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

    // write to output array
    if (i < N)
      Y[i] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x-1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;

}

// exclusive-scan of carries
__global__ void scan_kernel_2(int *carries)
{
  __shared__ int shared_buffer[256];

  // load data:
  int my_carry = carries[threadIdx.x];

  // exclusive scan in shared buffer:

  for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();

  // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
}

__global__ void scan_kernel_3(int *Y, int N,
                              int const *carries)
{
  unsigned int work_per_thread = (N - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ int shared_offset;

  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];

  __syncthreads();

  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < N)
      Y[i] += shared_offset;
}




void exclusive_scan(int const * input,
                    int       * output, int N)
{
  int num_blocks = 256;
  int threads_per_block = 256;

  int *carries;
  cudaMalloc(&carries, sizeof(int) * num_blocks);

  // First step: Scan within each thread group and write carries
  scan_kernel_1<<<num_blocks, threads_per_block>>>(input, output, N, carries);

  // Second step: Compute offset for each thread group (exclusive scan for each thread group)
  scan_kernel_2<<<1, num_blocks>>>(carries);

  // Third step: Offset each thread group accordingly
  scan_kernel_3<<<num_blocks, threads_per_block>>>(output, N, carries);

  cudaFree(carries);
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
  double *cuda_solution, *cuda_p, *cuda_r, *cuda_Ap, *cuda_scalar;
  cudaMalloc(&cuda_p, sizeof(double) * N);
  cudaMalloc(&cuda_r, sizeof(double) * N);
  cudaMalloc(&cuda_Ap, sizeof(double) * N);
  cudaMalloc(&cuda_solution, sizeof(double) * N);
  cudaMalloc(&cuda_scalar, sizeof(double));

  cudaMemcpy(cuda_p, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_r, rhs, sizeof(double) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(cuda_solution, solution, sizeof(double) * N, cudaMemcpyHostToDevice);

  const double zero = 0;
  double residual_norm_squared = 0;
  cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
  cuda_dot_product<<<512, 512>>>(N, cuda_r, cuda_r, cuda_scalar);
  cudaMemcpy(&residual_norm_squared, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);

  double initial_residual_squared = residual_norm_squared;

  int iters = 0;
  cudaDeviceSynchronize();
  timer.reset();
  while (1) {

    // line 4: A*p:
    cuda_csr_matvec_product<<<512, 512>>>(N, csr_rowoffsets, csr_colindices, csr_values, cuda_p, cuda_Ap);

    // lines 5,6:
    cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cuda_dot_product<<<512, 512>>>(N, cuda_p, cuda_Ap, cuda_scalar);
    cudaMemcpy(&alpha, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);
    alpha = residual_norm_squared / alpha;

    // line 7:
    cuda_vecadd<<<512, 512>>>(N, cuda_solution, cuda_p, alpha);

    // line 8:
    cuda_vecadd<<<512, 512>>>(N, cuda_r, cuda_Ap, -alpha);

    // line 9:
    beta = residual_norm_squared;
    cudaMemcpy(cuda_scalar, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cuda_dot_product<<<512, 512>>>(N, cuda_r, cuda_r, cuda_scalar);
    cudaMemcpy(&residual_norm_squared, cuda_scalar, sizeof(double), cudaMemcpyDeviceToHost);

    // line 10:
    if (std::sqrt(residual_norm_squared / initial_residual_squared) < 1e-6) {
      break;
    }

    // line 11:
    beta = residual_norm_squared / beta;

    // line 12:
    cuda_vecadd2<<<512, 512>>>(N, cuda_p, cuda_r, beta);

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
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
void solve_system(int N, int M) {

  // std::cout << "Solving Ax=b with " << N << " unknowns." << std::endl;

  //
  // Allocate CSR arrays.
  //
  // Note: Usually one does not know the number of nonzeros in the system matrix
  // a-priori.
  //       For this exercise, however, we know that there are at most 5 nonzeros
  //       per row in the system matrix, so we can allocate accordingly.
  //

  int nval = (5 * (N-2) * (M-2)) + (3 * 4)  + (4 * 2 * (N - 2)) + (4 * 2 * (M-2));
  int NM = N * M;

  int *csr_rowoffsets = (int *)malloc(sizeof(int) * (NM + 1));
  int *csr_colindices = (int *)malloc(sizeof(int) * nval);
  double *csr_values = (double *)malloc(sizeof(double) * nval);


  int *cuda_csr_rowoffsets, *cuda_csr_colindices, *cuda_nnz;
  double *cuda_csr_values;
  cudaMalloc(&cuda_csr_rowoffsets, sizeof(int) * (NM + 1));
  cudaMalloc(&cuda_nnz, sizeof(int) * (NM));
  cudaMalloc(&cuda_csr_colindices, sizeof(int) * nval);
  cudaMalloc(&cuda_csr_values, sizeof(double) * nval);



  cudaDeviceSynchronize();
  Timer timer;
  double time{0};

  timer.reset();

  count_nnz<<<256, 256>>>(cuda_nnz, N, M);
  exclusive_scan(cuda_nnz, cuda_csr_rowoffsets, (NM + 1));
  assembleA<<<256,256>>>(cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, N, M);

  cudaDeviceSynchronize();
  time = timer.get();
  std::cout << time;
  //
  // fill CSR matrix with values
  //

  //
  // Allocate solution vector and right hand side:
  //
  // double *solution = (double *)malloc(sizeof(double) * (NM));
  // double *rhs = (double *)malloc(sizeof(double) * (NM));
  // std::fill(rhs, rhs + (NM), 1);

  //
  // Allocate CUDA-arrays //
  //

  //
  // Call Conjugate Gradient implementation with GPU arrays
  //
  // conjugate_gradient(NM, cuda_csr_rowoffsets, cuda_csr_colindices, cuda_csr_values, rhs, solution);

  //
  // Check for convergence:
  //

  // cudaMemcpy(csr_rowoffsets, cuda_csr_rowoffsets, sizeof(int) * (NM + 1), cudaMemcpyDeviceToHost);
  // cudaMemcpy(csr_colindices, cuda_csr_colindices, sizeof(int) * nval, cudaMemcpyDeviceToHost);
  // cudaMemcpy(csr_values, cuda_csr_values, sizeof(double) * nval, cudaMemcpyDeviceToHost);
  // double residual_norm = relative_residual(NM, csr_rowoffsets, csr_colindices, csr_values, rhs, solution);
  // std::cout << "Relative residual norm: " << residual_norm
  //           << " (should be smaller than 1e-6)" << std::endl;

  cudaFree(cuda_csr_rowoffsets);
  cudaFree(cuda_csr_colindices);
  cudaFree(cuda_csr_values);
  cudaFree(cuda_nnz);
  // free(solution);
  // free(rhs);
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
