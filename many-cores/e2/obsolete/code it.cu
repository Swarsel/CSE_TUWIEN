#include <stdio.h>
#include "timer.hpp"
 
 
__global__
void saxpy(int n, float a, float *x, float *y)
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

  for(int j=0; j<=6;j++){
  
  float time1 =0.0;
  float time2 =0.0;
  float time3 =0.0;

  for(int it=0;it <= 10; it++){
  printf("N:%d \n", j);
  
  float *x, *y, *d_x, *d_y;
  Timer timer;

  
 
  // Allocate device memory and copy host data over
  timer.reset();
  cudaMalloc(&d_x, N[j]*sizeof(float)); 
  time1 += timer.get();
  

  cudaMalloc(&d_y, N[j]*sizeof(float));
 
  timer.reset();
  x = (float*)malloc(N[j]*sizeof(float));
  for (int i = 0; i < N[j]; i++) {
  x[i] = 1.0f;
  }
  cudaMemcpy(d_x, x, N[j]*sizeof(float), cudaMemcpyHostToDevice);
  time2 += timer.get();

  y = (float*)malloc(N[j]*sizeof(float));
  cudaMemcpy(d_y, y, N[j]*sizeof(float), cudaMemcpyHostToDevice);
 
  // tidy up host and device memory

  timer.reset();
  cudaFree(d_x);
  time3 += timer.get();
  
  cudaFree(d_y);
  free(x);
  free(y);
  }

  printf("timer Alloc: %f\n", time1/10);
  printf("timer copy: %f\n", time2/10);
  printf("timer Free: %f\n", time3/10);
  }
 
  return EXIT_SUCCESS;
}

