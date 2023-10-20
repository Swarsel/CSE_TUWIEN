#include <stdio.h>
#include "timer.hpp"


__global__ void prod(int n, double *x, double *y, double *z)
{
  __shared__ double shared_m[256];
  double thread_prod = 0;
  unsigned int total_threads = blockDim.x * gridDim.x;
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  for (unsigned int i = thread_id; i<n; i += total_threads) {
    thread_prod += x[i] * y[i];
  }
  shared_m[threadIdx.x] = thread_prod;
  for (unsigned int stride = blockDim.x/2; stride>0; stride/=2) {
    __syncthreads();
    if (threadIdx.x < stride) {
    shared_m[threadIdx.x] += shared_m[threadIdx.x + stride];
    }
  }
  if (threadIdx.x == 0) {
  z[blockIdx.x] = shared_m[0];
  }
}
int main(void)
{
  int N[7] = {100, 300, 1000, 10000, 100000, 1000000, 3000000};

  for(int j=0; j <= 6; j++){
    double sum = 0.0;
    double *x, *y, *z, *d_x, *d_y, *d_z;
    Timer timer;
    double time_sum = 0.0;

    for(int it=0; it < 10; it++){
      x = (double*)malloc(N[j]*sizeof(double));
      y = (double*)malloc(N[j]*sizeof(double));
      z = (double*)malloc(256*sizeof(double));
      cudaMalloc(&d_x, N[j]*sizeof(double));
      cudaMalloc(&d_y, N[j]*sizeof(double));
      cudaMalloc(&d_z, 256*sizeof(double));

      for (int i = 0; i < N[j]; i++) {
        x[i] = (double)i;
        y[i] = (double)(N[j]-i-1);
      }
      cudaMemcpy(d_x, x, N[j]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y, N[j]*sizeof(double), cudaMemcpyHostToDevice);

      timer.reset();
      prod<<<256, 256>>>(N[j], d_x, d_y, d_z);
      cudaDeviceSynchronize();
      cudaMemcpy(z, d_z, 256*sizeof(double), cudaMemcpyDeviceToHost);
      for(int i = 0; i < 256; i++) sum += z[i];
      time_sum += timer.get();

      cudaFree(d_x);
      cudaFree(d_y);
      cudaFree(d_z);
      free(x);
      free(y);
      free(z);
    }
    printf("%f\n", time_sum/10);
  }
  return EXIT_SUCCESS;
}
