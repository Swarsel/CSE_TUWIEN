#include "timer.hpp"
#include "cuda_errchk.hpp"
#include <algorithm>
#include <iostream>
#include <ostream>
#include <stdio.h>

// result = (x, y)
__global__ void task1_a(int N, double *x, double *result)
{
    __shared__ double shared_mem_sum[256];
    __shared__ double shared_mem_absum[256];
    __shared__ double shared_mem_sqsum[256];
    __shared__ double shared_mem_nn0[256];

    double sum = 0;
    double absum = 0;
    double sqsum = 0;
    double nn0 = 0;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
        sum += x[i];
        absum += abs(x[i]);
        sqsum += pow(x[i],2);
        if (x[i] == 0) nn0 += 1;
    }

    shared_mem_sum[threadIdx.x] = sum;
    shared_mem_absum[threadIdx.x] = absum;
    shared_mem_sqsum[threadIdx.x] = sqsum;
    shared_mem_nn0[threadIdx.x] = nn0;
    for (int k = blockDim.x / 2; k > 0; k /= 2) {
        __syncthreads();
        if (threadIdx.x < k) {
            shared_mem_sum[threadIdx.x] += shared_mem_sum[threadIdx.x + k];
            shared_mem_absum[threadIdx.x] += shared_mem_absum[threadIdx.x + k];
            shared_mem_sqsum[threadIdx.x] += shared_mem_sqsum[threadIdx.x + k];
            shared_mem_nn0[threadIdx.x] += shared_mem_nn0[threadIdx.x + k];
        }
    }

    if (threadIdx.x == 0) {
        atomicAdd(&result[0], shared_mem_sum[0]);
        atomicAdd(&result[1], shared_mem_absum[0]);
        atomicAdd(&result[2], shared_mem_sqsum[0]);
        atomicAdd(&result[3], shared_mem_nn0[0]);
    }
}

bool check(const double *result, int N) {
    if (result[0] == N * (-2)) {
        if (result[1] == N * 2) {
            if (result[2] == N * 4) {
                if (result[3] == 0) return 1;
                else std::cout << "nonzero wrong, is " << result[3] << " but should be 0 " << std::endl;
            }
            else std::cout << "square wrong, is " << result[2] << " but should be " << N * 4 << std::endl;
        }
        else std::cout << "abs wrong, is " << result[1] << " but should be " << N * 2 << std::endl;
    }
    else std::cout << "sum wrong, is " << result[0] << " but should be " << N * (-2) << std::endl;
    return 0;
}

/** Solve a system with `points_per_direction * points_per_direction` unknowns
 */
int main() {

    int N[9] = {100, 300, 1000, 10000, 100000, 1000000, 3000000, 10000000, 100000000};
    Timer timer;
    double time;
    std::cout << "[";
    //
    // Allocate and initialize arrays on CPU
    //
    for(int j=0; j <= 8; j++) {
        time = 0;
        double *x = (double *)malloc(sizeof(double) * N[j]);
        // double *y = (double *)malloc(sizeof(double) * N);

        std::fill(x, x + N[j], -2);
        // std::fill(y, y + N, 2);
        // for (int i = 0; i < N[j]; i++) {
        //     if (i % 2 == 0) x[i] = -2;
        //     else x[i] = 0;
        // }

        double *cuda_x;
        //double *cuda_y;
        double *cuda_results;
        CUDA_ERRCHK(cudaMalloc(&cuda_x, sizeof(double) * N[j]));
        //CUDA_ERRCHK(cudaMalloc(&cuda_y, sizeof(double) * N));
        CUDA_ERRCHK(cudaMalloc(&cuda_results, 4 * sizeof(double)));

        for(int it=0; it < 10; it++){
            //
            // Allocate and initialize arrays on GPU
            //
            double results[4] = {0, 0, 0, 0};

            CUDA_ERRCHK(cudaMemcpy(cuda_x, x, sizeof(double) * N[j], cudaMemcpyHostToDevice));
            // CUDA_ERRCHK(cudaMemcpy(cuda_y, y, sizeof(double) * N, cudaMemcpyHostToDevice));
            CUDA_ERRCHK(cudaMemcpy(cuda_results, &results, 4 * sizeof(double), cudaMemcpyHostToDevice));
            timer.reset();
            task1_a<<<256, 256>>>(N[j], cuda_x, cuda_results);
            time += timer.get();
            CUDA_ERRCHK(cudaMemcpy(&results, cuda_results, 4 * sizeof(double), cudaMemcpyDeviceToHost));
            check(results, N[j]);
            //
            // Clean up
            //
            // free(y);
        }

        if (j != 8) std::cout << time/10 << ", ";
        // std::cout << "Result of dot product: " << results << std::endl;
        CUDA_ERRCHK(cudaFree(cuda_x));
        //CUDA_ERRCHK(cudaFree(cuda_y));
        CUDA_ERRCHK(cudaFree(cuda_results));
        free(x);
    }
    std::cout << time/10 << "]";
    return EXIT_SUCCESS;
}
