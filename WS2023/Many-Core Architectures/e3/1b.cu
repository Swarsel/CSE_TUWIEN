#include <stdio.h>
#include "timer.hpp"
#include "vector"
#include <algorithm>
#include <iostream>

__global__ void skip(int n, int k, double *x, double *y, double *z)
{
	unsigned int total_threads = blockDim.x * gridDim.x;
	int thread_id = blockIdx.x*blockDim.x + threadIdx.x;

    for (unsigned int i = thread_id; i <= n-k; i += total_threads) z[i+k] = x[i+k] + y[i+k];
}

int main(void)
{
  std::cout << "[";
  int N = 100000000;
  double *x, *y, *z, *d_x, *d_y, *d_z, time, bw;
  Timer timer;

  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));
  z = (double*)malloc(N*sizeof(double));

  for (int i = 0; i < N; i++) {
    x[i] = i;
    y[i] = N-i-1;
  }

  cudaMalloc(&d_x, N*sizeof(double));
  cudaMalloc(&d_y, N*sizeof(double));
  cudaMalloc(&d_z, N*sizeof(double));
  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

  for (int k = 0; k < 64; k++) {
      time = 0;

	  for (int i = 0; i < 10; i++) {
	   	  timer.reset();
	      skip<<<256, 256>>>(N, k, d_x, d_y, d_z);
		  cudaDeviceSynchronize();
	  	  time += timer.get();
	  }
	  time /= 10;
      bw = 3 * (N-k) * sizeof(double) / (1e9 * time);
	  if (k!=63) std::cout << bw << ", ";
      if (k==63) std::cout << bw << "]";

}

  cudaMemcpy(z, d_z, N*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_z);
  free(x);
  free(y);
  free(z);

  cudaDeviceReset();
  return EXIT_SUCCESS;
}
