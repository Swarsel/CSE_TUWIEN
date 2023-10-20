#include <stdio.h>
#include "timer.hpp"


__global__ void add(int n, double *x, double *y, double *z)
{
  unsigned int total_threads =  blockDim.x * gridDim.x;
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  for (int i = thread_id; i<n; i += total_threads) z[i] = x[i] + y[i];
}

int main(void)
{
  int N = 10000000;
  int gridSize[7] = {16, 32, 64, 128, 256, 512, 1024};
  int blockSize[7] = {16, 32, 64, 128, 256, 512, 1024};

  double *x, *y, *d_x, *d_y;
  x = (double*)malloc(N*sizeof(double));
  y = (double*)malloc(N*sizeof(double));
  cudaMalloc(&d_x, N*sizeof(double));
  cudaMalloc(&d_y, N*sizeof(double));

  for (int i = 0; i < N; i++) {
    x[i] = (double)i;
    y[i] = (double)(N-i-1);
  }
  cudaMemcpy(d_x, x, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(double), cudaMemcpyHostToDevice);

  for(int ii=0; ii < 7; ii++){
    printf("%d & ",gridSize[ii]);
    for(int jj=0; jj < 7; jj++){
      double time_sum = 0.0;
      for(int it=0; it<5;it++){
        double *z, *d_z;
        Timer timer;
        z = (double*)malloc(N*sizeof(double));
        cudaMalloc(&d_z, N*sizeof(double));

        cudaDeviceSynchronize();
        timer.reset();
        add<<<blockSize[jj], gridSize[ii]>>>(N, d_x, d_y, d_z);
        cudaDeviceSynchronize();
        time_sum += timer.get();

        cudaFree(d_z);
        free(z);
        }
      printf("%f & ", time_sum/5);
    }
  printf("\\\\ \n");
  }
  return EXIT_SUCCESS;
}
