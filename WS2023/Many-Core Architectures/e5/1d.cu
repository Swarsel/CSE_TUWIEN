#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>

__global__
void copyKernel(double *dest, int it){
    for (unsigned int i = 0; i<it; i ++) {
        atomicAdd(dest,1);
    }
}

int main(int argc, char *argv[]) {
    int blocks = std::atoi(argv[1]);
    int threads = std::atoi(argv[2]);
    Timer timer;
    double time = 0;
    // double bw;
    // double *x, *y, *cuda_x, *cuda_y;
    double *cuda_x;

    // x = (double*)malloc(sizeof(double) * N);
    cudaMalloc(&cuda_x, sizeof(double));
    // cudaMalloc(&cuda_y, sizeof(double));

//     for ( int i = 0; i < N ; i++) {
//         // i know it is bad practise to use rand(), but the numbers do not really need to be random here
//         // i just want to minimize the chance of some compiler optimization happening
//         x[i] = (rand() % 100);
// }

    // cudaMemcpy(cuda_x, x, sizeof(double) * N, cudaMemcpyHostToDevice);
    cudaMemset(cuda_x, 0, 1);
    // cudaDeviceSynchronize();
    // for(int it = 0; it < 10; it++){
    int it = 1000;
        timer.reset();
        copyKernel<<<blocks,threads>>>(cuda_x, it);
        cudaDeviceSynchronize();
        // time += timer.get();
        time = timer.get();
        // cudaMemset(cuda_y, 0, N);
    // }
    // time /= 10;
    // time = 10;
        // int worked = blocks * threads * it
    // bw = sizeof(double) *2 * N / time / 1e9;
    // std::cout << bw;
        std::cout << time;

    // free(x);
    // free(y);
    cudaFree(cuda_x);
    // cudaFree(cuda_y);


    return EXIT_SUCCESS;
}
