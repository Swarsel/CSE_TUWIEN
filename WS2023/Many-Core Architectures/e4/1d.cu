#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <stdio.h>

// result = (x, y)
__global__ void task1_d(int N, double *x, double *y, double *result)
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

bool check(const double& result, int N) {
    if (result == N * 2) return 1;
    else std::cout << "dotproduct wrong, is " << result << " but should be " << N * 2 << std::endl;
    return 0;
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
int main() {

  int N[7] = {100, 300, 1000, 10000, 100000, 1000000, 3000000};
  Timer timer;
  double time;
  std::cout << "[";
 //
  // Allocate and initialize arrays on CPU
  //
  for(int j=0; j <= 6; j++) {
      time = 0;
      double *x = (double *)malloc(sizeof(double) * N[j]);
      double *y = (double *)malloc(sizeof(double) * N[j]);

      std::fill(x, x + N[j], 1);
      std::fill(y, y + N[j], 2);
      // for (int i = 0; i < N[j]; i++) {
      //     if (i % 2 == 0) x[i] = -2;
      //     else x[i] = 0;
      // }

      double *cuda_x;
      double *cuda_y;
      double *cuda_results;
      CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N[j]));
      CUDA_ERRCHK(cudaMalloc(&cuda_y, sizeof(double) * N[j]));
      CUDA_ERRCHK(cudaMalloc(&cuda_results, sizeof(double)));

    for(int it=0; it < 10; it++){
        //
        // Allocate and initialize arrays on GPU
        //
        double results = 0;

        CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N[j], cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaMemcpy(cuda_y, y, sizeof(double) * N[j], cudaMemcpyHostToDevice));
        CUDA_ERRCHK(cudaMemcpy(cuda_results, &results, sizeof(double), cudaMemcpyHostToDevice));
        timer.reset();
        task1_d<<<256/256, 256>>>(N[j], cuda_x, cuda_y, cuda_results);
        time += timer.get();
        CUDA_ERRCHK(cudaMemcpy(&results, cuda_results, sizeof(double), cudaMemcpyDeviceToHost));
        check(results, N[j]);

        //
        // Clean up
        //
        // free(y);
    }

    if (j != 6) std::cout << time/10 << ", ";
    // std::cout << "Result of dot product: " << results << std::endl;
    CUDA_ERRCHK(cudaFree(cuda_x));
    //CUDA_ERRCHK(cudaFree(cuda_y));
    CUDA_ERRCHK(cudaFree(cuda_results));
    free(x);
    free(y);
  }
  std::cout << time/10 << "]";
  return EXIT_SUCCESS;
}
