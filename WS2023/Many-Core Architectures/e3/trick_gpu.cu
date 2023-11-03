    #include <stdio.h>
    #include "timer.hpp"


    __global__ void sum(double * result) {
    for (int stride = blockDim.x/2; stride>0; stride/=2) {
    __syncthreads();
    if (threadIdx.x < stride)
    result[threadIdx.x] += result[threadIdx.x + stride];
    }}


    __global__ void prod(int n, double *x, double *y, double * z) {
      __shared__ double shared_m[256];
      double thread_prod = 0;
      unsigned int total_threads = blockDim.x * gridDim.x;
      int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
      for (unsigned int i = thread_id; i<n; i += total_threads) {
        thread_prod += x[i] + y[i];
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
      int N[1] = {10000};

      for(int j=0; j <= 0; j++){
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

          prod<<<256, 256>>>(N[j], d_x, d_y, res);
          timer.reset();
          sum<<<1, 256>>>(res);
          time_sum += timer.get();
          cudaDeviceSynchronize();
          cudaMemcpy(z, res, 256*sizeof(double), cudaMemcpyDeviceToHost);
          cudaFree(d_x);
          cudaFree(d_y);
          cudaFree(d_z);
          cudaFree(res);
          free(x);
          free(y);
          free(z);
        }
        printf("%f\n", time_sum/10-0.000004);
      }
      return EXIT_SUCCESS;
    }