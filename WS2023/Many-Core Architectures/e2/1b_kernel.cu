#include <stdio.h>
#include "timer.hpp"

__global__ void init(double *x, double *y, int N) {
	unsigned int total_threads = blockDim.x * gridDim.x;
	int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = thread_id; i<N; i += total_threads) {
		x[i] = (double)i;
		y[i] = (double)(N-i-1);
	}
}

int main(void)
{
  int N[7] = {100, 300, 1000, 10000, 100000, 1000000, 3000000};

  for(int j=0; j <= 6; j++){
    double *x, *y, *d_x, *d_y;
    Timer timer;
    double time = 0.0;
    double time_sum = 0.0;

    for(int it=0; it<10; it++){
      cudaDeviceSynchronize();

      timer.reset();
      x = (double*)malloc(N[j]*sizeof(double));
	  y = (double*)malloc(N[j]*sizeof(double));
      cudaMalloc(&d_x, N[j]*sizeof(double));
      cudaMalloc(&d_y, N[j]*sizeof(double));
      init<<<256, 256>>>(d_x, d_y, N[j]);
      cudaDeviceSynchronize();
      time = timer.get();

      time_sum += time;
      cudaFree(d_x);
      }
    printf("%f\n", time_sum/10);
  }
  return EXIT_SUCCESS;
}
