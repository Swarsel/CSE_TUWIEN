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
  int N[7];
  N[0] = 100;
  N[1] = 300;
  N[2] = 1000;
  N[3] = 10000;
  N[4] = 100000;
  N[5] = 1000000;
  N[6] = 3000000;
  N[7] = 0;

  for(int j=0; j < 7; j++){ 
  double *x, *y, *d_x, *d_y;
  Timer timer;
  double time = 0.0;
  double time_sum = 0.0;

  for(int it=0; it<10;it++){
 
  // Allocate device memory and copy host data over
  cudaDeviceSynchronize();
  timer.reset();
  cudaMalloc(&d_x, N[j]*sizeof(double)); 
 
  time = timer.get();
  cudaDeviceSynchronize();
  time_sum += time;
  cudaFree(d_x);
  }
  printf("%f\n", time_sum/10);
}
  return EXIT_SUCCESS;
}
 