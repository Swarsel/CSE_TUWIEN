#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <iostream>
#include <string>

__global__
void copyKernel(int N, double src1, double src2){
  volatile double res = 0;
  for(int i = 0; i < 1000; i ++){
    res += src1 * src2;
  }
}

__global__
void refKernel(int N, double src1, double src2){
  volatile double res = 0;
  for(int i = 0; i < 1000; i ++){
    ;
  }
}
int main(int argc, char *argv[]) {
    // int blocks = std::atoi(argv[1]);
    // int threads = std::atoi(argv[2]);
    int N = std::stoi(argv[1]);
    Timer timer;
    double time = 0;
    double time2 = 0;
    double x = rand();
    double y = rand();
        copyKernel<<<4096,1024>>>(N, x, y);
        timer.reset();
        cudaDeviceSynchronize();
        time = timer.get();

        timer.reset();
        refKernel<<<4096,1024>>>(N, x, y);
        cudaDeviceSynchronize();
        time2 = timer.get();

        std::cout << time << " " << time2 ;

    // free(x);
    // free(y);
    // cudaFree(cuda_x);
    // cudaFree(cuda_y);


    return EXIT_SUCCESS;
}
