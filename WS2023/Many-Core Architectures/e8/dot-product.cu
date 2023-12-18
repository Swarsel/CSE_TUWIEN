#include "cuda_errchk.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cuda_runtime_api.h>
#include <iostream>
#include <stdio.h>

// result = (x, y)
__global__ void cuda_dot_product(int N, double *x, double *y, double *result) {
  __shared__ double shared_mem[128];

  double dot = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N;
       i += blockDim.x * gridDim.x) {
    dot += x[i] * y[i];
  }

  shared_mem[threadIdx.x] = dot;
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    __syncthreads();
    if (threadIdx.x < k) {
      shared_mem[threadIdx.x] += shared_mem[threadIdx.x + k];
    }
  }

  if (threadIdx.x == 0)
    atomicAdd(result, shared_mem[0]);
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
int main(int argc, char *argv[]) {
  int N = std::atoi(argv[1]);
  //
  // Allocate and initialize arrays on CPU
  //
  double *x = (double *)malloc(sizeof(double) * N);
  double *y = (double *)malloc(sizeof(double) * N);
  double alpha = 0;

  std::fill(x, x + N, 1);
  std::fill(y, y + N, 2);

  //
  // Allocate and initialize arrays on GPU
  //
  double *cuda_x;
  double *cuda_y;
  double *cuda_alpha;

  CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_y, sizeof(double) * N));
  CUDA_ERRCHK(cudaMalloc(&cuda_alpha, sizeof(double)));

  CUDA_ERRCHK(
      cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(
      cudaMemcpy(cuda_y, y, sizeof(double) * N, cudaMemcpyHostToDevice));
  CUDA_ERRCHK(
      cudaMemcpy(cuda_alpha, &alpha, sizeof(double), cudaMemcpyHostToDevice));
  Timer timer;
  double time;
  timer.reset();
  cuda_dot_product<<<128, 128>>>(N, cuda_x, cuda_y, cuda_alpha);
  cudaDeviceSynchronize();
  time = timer.get();
  std::cout << time;
  CUDA_ERRCHK(
      cudaMemcpy(&alpha, cuda_alpha, sizeof(double), cudaMemcpyDeviceToHost));

  // std::cout << "Result of dot product: " << alpha << std::endl;

  //
  // Clean up
  //
  CUDA_ERRCHK(cudaFree(cuda_x));
  CUDA_ERRCHK(cudaFree(cuda_y));
  CUDA_ERRCHK(cudaFree(cuda_alpha));
  free(x);
  free(y);

  return EXIT_SUCCESS;
}
