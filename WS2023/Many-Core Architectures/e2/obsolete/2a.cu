#include <stdio.h>
#include "timer.hpp"


__device__ void sum(double * shared_m, double * result) {

for (int stride = blockDim.x/2; stride>0; stride/=2) {
__syncthreads();
if (threadIdx.x < stride)
shared_m[threadIdx.x] += shared_m[threadIdx.x + stride];
}
if (threadIdx.x == 0) {
result[blockIdx.x] = shared_m[0];
}}

__global__ void dot_product(int N, double *x, double *y, double * result) {

__shared__ double shared_m[256];
double thread_sum = 0;
int threads_step = 256;
int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
for (int i = thread_id; i<N; i += threads_step)
thread_sum += x[i] * y[i];

shared_m[threadIdx.x] = thread_sum;
sum(shared_m, result);
}
int main(void)
{
  int N[7] = {100, 300, 1000, 10000, 100000, 1000000, 3000000};

  for(int j=0; j <= 6; j++){
    double *res;
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
      cudaMalloc(&res, N[j]*sizeof(double));

      for (int i = 0; i < N[j]; i++) {
        x[i] = (double)i;
        y[i] = (double)(N[j]-i-1);
      }
      cudaMemcpy(d_x, x, N[j]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y, N[j]*sizeof(double), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      timer.reset();
      dot_product<<<256, 256>>>(N[j], d_x, d_y, res);
      cudaDeviceSynchronize();
      cudaMemcpy(z, res, N[j]*sizeof(float), cudaMemcpyDeviceToHost);
      time_sum += timer.get();
      cudaFree(d_x);
      cudaFree(d_y);
      cudaFree(d_z);
      cudaFree(res);
      free(x);
      free(y);
      free(z);
    }
    printf("%f\n", time_sum/10);
  }
  return EXIT_SUCCESS;
}
