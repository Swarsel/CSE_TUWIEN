#include <cstddef>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include "timer.hpp"
int main(int argc, char **argv)
{

    cublasHandle_t h;
    cublasCreate_v2(&h);
    double time=0;
    Timer timer;
    const size_t N = atoi(argv[1]);
    const size_t K = atoi(argv[2]);
    // for (int ni = 0; ni <= 0; ni++) {

        //
        // Initialize CUBLAS:
        //
        // std::cout << "Init CUBLAS..." << std::endl;


        //
        // allocate host memory:
        //
        // std::cout << "Allocating host arrays..." << std::endl;
        double  *x = (double*)malloc(sizeof(double) * N);
        double **y = (double**)malloc(sizeof(double*) * K);
        for (size_t i=0; i<K; ++i) {
            y[i] = (double*)malloc(sizeof(double) * N);
        }
        double *results  = (double*)malloc(sizeof(double) * K);
        double *results2 = (double*)malloc(sizeof(double) * K);


        //
        // allocate device memory
        //
        // std::cout << "Allocating CUDA arrays..." << std::endl;
        double *cuda_x; cudaMalloc( (void **)(&cuda_x), sizeof(double)*N);
        double **cuda_y = (double**)malloc(sizeof(double*) * K);  // storing CUDA pointers on host!
        for (size_t i=0; i<K; ++i) {
            cudaMalloc( (void **)(&cuda_y[i]), sizeof(double)*N);
        }

        //
        // fill host arrays with values
        //
        for (size_t j=0; j<N; ++j) {
            x[j] = 1 + j%K;
        }
        for (size_t i=0; i<K; ++i) {
            for (size_t j=0; j<N; ++j) {
                y[i][j] = 1 + rand() / (1.1 * RAND_MAX);
            }
        }

        //
        // Reference calculation on CPU:
        //
        for (size_t i=0; i<K; ++i) {
            results[i] = 0;
            results2[i] = 0;
            for (size_t j=0; j<N; ++j) {
                results[i] += x[j] * y[i][j];
            }
        }

        //
        // Copy data to GPU
        //
        // std::cout << "Copying data to GPU..." << std::endl;
        cudaMemcpy(cuda_x, x, sizeof(double)*N, cudaMemcpyHostToDevice);
        for (size_t i=0; i<K; ++i) {
            cudaMemcpy(cuda_y[i], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
        }


        for(int it=0; it < 10; it++) {
            std::fill(results2, results2 + K, 0);
            timer.reset();
            for (size_t i=0; i<K; ++i) {
                cublasDdot(h, N, cuda_x, 1, cuda_y[i], 1, results2 + i);
            }
            cudaDeviceSynchronize();

            time += timer.get();
        }
        time /= 10;
        std::cout << time;
        //else if (N == Ns[0]) std::cout << time << "]" << std::endl;
        //
        // Let CUBLAS do the work:
        //
        // std::cout << "Running dot products with CUBLAS..." << std::endl;

        //
        // Compare results
        //
        // std::cout << "Copying results back to host..." << std::endl;
        // for (size_t i=0; i<K; ++i) {
        //     std::cout << results[i] << " on CPU, " << results2[i] << " on GPU. Relative difference: " << fabs(results[i] - results2[i]) / results[i] << std::endl;
        // }


        //
        // Clean up:
        //
        // std::cout << "Cleaning up..." << std::endl;
        free(x);
        cudaFree(cuda_x);

        for (size_t i=0; i<K; ++i) {
            free(y[i]);
            cudaFree(cuda_y[i]);
        }
        free(y);
        free(cuda_y);

        free(results);
        free(results2);

    // }
        cublasDestroy(h);
    return 0;
}