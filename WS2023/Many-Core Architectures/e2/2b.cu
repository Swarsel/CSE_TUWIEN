#include <stdio.h>
#include "timer.hpp"


__global__ void prod(int n, double *x, double *y, double *z)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) z[i] = x[i] * y[i];
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
      z = (double*)malloc(N[j]*sizeof(double));
      cudaMalloc(&d_x, N[j]*sizeof(double));
      cudaMalloc(&d_y, N[j]*sizeof(double));
      cudaMalloc(&d_z, N[j]*sizeof(double));

      for (int i = 0; i < N[j]; i++) {
        x[i] = (double)i;
        y[i] = (double)(N[j]-i-1);
      }
      cudaMemcpy(d_x, x, N[j]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y, N[j]*sizeof(double), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      timer.reset();
      prod<<<256, 256>>>(N[j], d_x, d_y, d_z);
      cudaDeviceSynchronize();
      cudaMemcpy(z, d_z, N[j]*sizeof(float), cudaMemcpyDeviceToHost);
      for(int i = 0; i < N[j]; i++) sum += z[i];
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
