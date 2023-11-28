#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <iostream>

int main(void) {
        double *x, *cuda_x;
        Timer timer;
        double time_sum = 0.0;
        x = (double*)malloc(1*sizeof(double));
        cudaMalloc(&cuda_x, 1*sizeof(double));
        x[0] = 0;
        cudaMemset(cuda_x, 1, 1);
        for (int it=0; it < 10000; it++) cudaMemcpy(cuda_x, x, sizeof(double), cudaMemcpyHostToDevice);

        timer.reset();
        cudaMemcpy(cuda_x, x, sizeof(double), cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        time_sum = timer.get();
        std::cout << time_sum;
        exit(0);
}