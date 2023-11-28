#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>

__global__ void kernelLaunch() {
    volatile int a = 1;
}


int main(void) {
        Timer timer;
        double time_sum;
        double *cuda_x;

        cudaMalloc(&cuda_x, 1000 * sizeof(double));
        cudaDeviceSynchronize();

        for (int it = 0; it <10000; it++) kernelLaunch<<<1,1>>> ();
        cudaDeviceSynchronize();
        timer.reset();
        kernelLaunch<<<1,1>>> ();
        time_sum = timer.get();

        std::cout << time_sum;

        cudaFree(cuda_x);
        exit(0);
}