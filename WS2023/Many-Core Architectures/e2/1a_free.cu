#include <stdio.h>
#include "timer.hpp"


__global__
void saxpy(int n, double a, double *x, double *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

int main(void)
{
  int N[7] = {100, 300, 1000, 10000, 100000, 1000000, 3000000};

  for(int j=0; j <=6 ; j++){
    double *x, *y, *d_x, *d_y;
    Timer timer;
    double time = 0.0;
    double time_sum = 0.0;

    for(int it=0; it<10;it++){
      cudaDeviceSynchronize();
      cudaMalloc(&d_x, N[j]*sizeof(double));
      cudaDeviceSynchronize();
      timer.reset();
      cudaFree(d_x);
      time = timer.get();
      time_sum += time;
  }
  printf("%f\n", time_sum/10);
}
  return EXIT_SUCCESS;
}
