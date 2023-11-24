#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <cuda_runtime_api.h>
#include <sm_60_atomic_functions.hpp>
#include <stdio.h>
#include <iostream>

__global__
void copyKernel(double *dest){
    for (unsigned int i = 0; i<100000; i ++) {
        atomicAdd(dest,1);
    }
}

__global__
void refKernel(int N, double *src, double *dest){
    for (unsigned int i = 0; i<100000; i ++) {
        ;
    }
}

int main(int argc, char *argv[]) {
    int N = std::atoi(argv[1]);
    Timer timer;
    double time = 0;
    double time2 = 0;
    double bw;
    double *x, *y, *cuda_x, *cuda_y;

    x = (double*)malloc(sizeof(double) * N);
    cudaMalloc(&cuda_x, sizeof(double) * N);
    cudaMalloc(&cuda_y, sizeof(double));

    for ( int i = 0; i < N ; i++) {
        // i know it is bad practise to use rand(), but the numbers do not really need to be random here
        // i just want to minimize the chance of some compiler optimization happening
        x[i] = (rand() % 100);
}

    cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();
    // for(int it = 0; it < 10; it++){
        timer.reset();
        copyKernel<<<256,256>>>(N, cuda_x, cuda_y);
        cudaDeviceSynchronize();
        // time += timer.get();
        time = timer.get();
        // in order to get a better reading of the bandwidth, run the same kernel again without the part that
        // we are interested in
        timer.reset();
        refKernel<<<256,256>>>(N, cuda_x, cuda_y);
        cudaDeviceSynchronize();
        time2 = timer.get();
        // cudaMemset(cuda_y, 0, N);
    // }
    // time /= 10;
    // time = 10;
    // bw = sizeof(double) *2 * N / time / 1e9;
    // std::cout << bw;
        std::cout << time << " " << time2;

    free(x);
    free(y);
    cudaFree(cuda_x);
    cudaFree(cuda_y);


    return EXIT_SUCCESS;
}
