#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <iostream>

__global__ void kernelLaunch() {
    volatile int a;
  }


int main(void) {
        Timer timer;
        double time_sum = 0.0;

        // for(int it=0; it < 10; it++){
            timer.reset();
            kernelLaunch<<<256,256>>> ();
            cudaDeviceSynchronize();
            // time_sum += timer.get();
            time_sum = timer.get();
        // }
        // i tested for refernce the time needed to allocate the volatile int on the CPU - measured time 0. i hope its similar on the GPU
        // timer.reset();
        // volatile int a;
        // double time = timer.get();
        // std::cout << time << std::endl;
        // std::cout << time_sum/10;
        std::cout << time_sum;
        exit(0);
}