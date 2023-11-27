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

        timer.reset();
        kernelLaunch<<<1,1>>> ();
        time_sum = timer.get();

        std::cout << time_sum;
        exit(0);
}