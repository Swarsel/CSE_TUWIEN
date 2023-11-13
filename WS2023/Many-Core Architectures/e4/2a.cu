#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <ostream>
#include <stdio.h>
#include <cmath>
#include <iostream>
#include "timer.hpp"
#include <cuda_errchk.hpp>

__global__ void dot(double *x, double **y, double *results, int i, int N) {
    double alpha1{0}, alpha2{0}, alpha3{0}, alpha4{0}, alpha5{0}, alpha6{0}, alpha7{0}, alpha8{0};

    for(int j = blockIdx.x * blockDim.x + threadIdx.x; j < N; j += blockDim.x*gridDim.x) {
        double val_w = x[j];
        alpha1 += val_w * y[i][j];
        alpha2 += val_w * y[i+1][j];
        alpha3 += val_w * y[i+2][j];
        alpha4 += val_w * y[i+3][j];
        alpha5 += val_w * y[i+4][j];
        alpha6 += val_w * y[i+5][j];
        alpha7 += val_w * y[i+6][j];
        alpha8 += val_w * y[i+7][j];
    }

    for (int j=16; j>0; j=j/2) {
        alpha1 += __shfl_xor_sync(0xffffffff, alpha1, j);
        alpha2 += __shfl_xor_sync(0xffffffff, alpha2, j);
        alpha3 += __shfl_xor_sync(0xffffffff, alpha3, j);
        alpha4 += __shfl_xor_sync(0xffffffff, alpha4, j);
        alpha5 += __shfl_xor_sync(0xffffffff, alpha5, j);
        alpha6 += __shfl_xor_sync(0xffffffff, alpha6, j);
        alpha7 += __shfl_xor_sync(0xffffffff, alpha7, j);
        alpha8 += __shfl_xor_sync(0xffffffff, alpha8, j);
    }

    if (threadIdx.x % 32 == 0) {
        atomicAdd(&results[i], alpha1);
        atomicAdd(&results[i+1], alpha2);
        atomicAdd(&results[i+2], alpha3);
        atomicAdd(&results[i+3], alpha4);
        atomicAdd(&results[i+4], alpha5);
        atomicAdd(&results[i+5], alpha6);
        atomicAdd(&results[i+6], alpha7);
        atomicAdd(&results[i+7], alpha8);
    }
}

int main(int argc, char *argv[])
{
    double time;
    // const size_t N = 100000;
    // const size_t K = 16;
    Timer timer;
    // int Ns[5] = {10000,100000,1000000,10000000,20000000};
    const int N = std::atoi(argv[1]);
    const int K = std::atoi(argv[2]);
    // int Ks[4] = {8,16,24,32};
    // int K = std::atoi(argv[1]);
    // for (int ki=0; ki<=3; ki++) {
        // int K = Ks[ki];
        // std::cout << "K " << K << std::endl;
        // std::cout << "[";

    // for (int ni = 0; ni <= 4; ni++) {
            // int N = Ns[ni];

            //
            // Initialize CUBLAS:
            //
            // std::cout << "Init CUBLAS..." << std::endl;
            // cublasHandle_t h;
            // cublasCreate(&h);


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
            double *cuda_x; cudaMalloc(&cuda_x, sizeof(double)*N);
            double *cuda_results; cudaMalloc(&cuda_results, sizeof(double)*K);
            double **cuda_y; cudaMalloc(&cuda_y, sizeof(double*) * K);
            double **y_pointers = (double**)malloc(sizeof(double*) * K);
            for (size_t i=0; i<K; ++i) {
                cudaMalloc(&y_pointers[i], sizeof(double)*N);
            }
            cudaMemcpy(cuda_y,y_pointers,sizeof(double*) * K, cudaMemcpyHostToDevice);

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
                cudaMemcpy(y_pointers[i], y[i], sizeof(double)*N, cudaMemcpyHostToDevice);
            }
            time = 0;
            std::fill(results2, results2 + K, 0);
            for(int it=0; it < 10; it++) {
                cudaMemcpy(cuda_results, results2, sizeof(double)*K, cudaMemcpyHostToDevice);
                timer.reset();
                for (int i=0; i<=(K-8); i += 8) dot<<<256, 256>>>(cuda_x, cuda_y, cuda_results, i, N);
                CUDA_ERRCHK(cudaDeviceSynchronize());
                time += timer.get();
            }
            time /= 10;
            std::cout << time;

            // if (N != Ns[4]) std::cout << time << ", ";
            // else if (N == Ns[4]) std::cout << time << "]" << std::endl;

            cudaMemcpy(results2, cuda_results, sizeof(double) * K, cudaMemcpyDeviceToHost);
            // Compare results
            //
            // std::cout << "Copying results back to host..." << std::endl;
            for (size_t i=0; i<K; ++i) {
                if (fabs(results[i] - results2[i]) / results[i] > 1e-10) {
                    std::cout << std::endl << "ATTENTION WRONG RESULT:" << results[i] << " on CPU, " << results2[i] << " on GPU. Relative difference: " << fabs(results[i] - results2[i]) / results[i] << std::endl;
                    return 1;
                }
            }

            //
            // Clean up:
            //
            // std::cout << "Cleaning up..." << std::endl;
            free(x);
            cudaFree(cuda_x);

            for (size_t i=0; i<K; ++i) {
                free(y[i]);
                cudaFree(y_pointers[i]);
            }
            free(y);
            cudaFree(cuda_y);
            cudaFree(cuda_results);

            free(results);
            free(results2);
            free(y_pointers);

            // cublasDestroy(h);
        // }
    // }

    return 0;
}
