#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <iostream>

__global__
void copyKernel(int N, double *src, double *dest){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N) {dest[id] = src[id];}
}

__global__
void refKernel(int N, double *src, double *dest){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if(id < N){;}
}

int main(int argc, char *argv[]) {
    int N = std::atoi(argv[1]);
    Timer timer;
    double time = 0;
    double bw;
    double *x, *y, *cuda_x, *cuda_y;

    x = (double*)malloc(sizeof(double) * N);
    y = (double*)malloc(sizeof(double) * N);
    cudaMalloc(&cuda_x, sizeof(double) * N);
    cudaMalloc(&cuda_y, sizeof(double) * N);

    memset(x, 1, N);

    cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    for(int it = 0; it < 10; it++){
        timer.reset();
        copyKernel<<<N+255/256,256>>>(N, cuda_x, cuda_y);
        cudaDeviceSynchronize();
        time += timer.get();
        timer.reset();
        refKernel<<<N+255/256,256>>>(N, cuda_x, cuda_y);
        cudaDeviceSynchronize();
        time -= timer.get();
        cudaMemset(cuda_y, 0, N);
    }

    time /= 10;
    bw = sizeof(double) *2 * N / time / 1e9;
    std::cout << bw;

    free(x);
    free(y);
    cudaFree(cuda_x);
    cudaFree(cuda_y);


    return EXIT_SUCCESS;
}
