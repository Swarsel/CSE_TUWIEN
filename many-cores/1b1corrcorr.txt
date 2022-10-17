#include <stdio.h>
#include "timer.hpp"
 
 
__global__
void init(int n, double *x)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) x[i] = 1.0f;
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
  double *x, *d_x;
  Timer timer;
  double time = 0.0;
  double time_sum = 0.0;

  for(int it=0; it<10;it++){
   x = (double*)malloc(N[j]*sizeof(double));
  cudaMalloc(&d_x, N[j]*sizeof(double)); 
  // Allocate device memory and copy host data over
  cudaDeviceSynchronize();
  timer.reset();
  init<<<1024, 256>>>(N[j], d_x);
  cudaMemcpy(d_x, x, N[j]*sizeof(double), cudaMemcpyHostToDevice);
 
  time = timer.get();
  cudaDeviceSynchronize();
  time_sum += time;
  free(x);
  cudaFree(d_x);
  }
  printf("%f\n", time_sum/10);
}
  return EXIT_SUCCESS;
}
 