#include <stdio.h>
#include <iostream>
#include "timer.hpp"
#include "cuda_errchk.hpp"   // for error checking of CUDA calls

__global__
void transpose(double *A, int N)
{
  int thread_id = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int total_threads = blockDim.x * gridDim.x;

  for (unsigned int i = thread_id; i < N*N; i += total_threads) {
      int r = i / N;
      int c = i % N;

      if (r < c) {
         double temp = A[N * r + c];
         A[N * r + c] = A[N * c + r];
         A[N * c + r] = temp;
      }
   }
}


void print_A(double *A, int N)
{
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; ++j) {
      std::cout << A[i * N + j] << ", ";
    }
    std::cout << std::endl;
  }
}

int main(void)
{
  double bw;
  std::cout << "[";
  for (int n=6; n<13;n++) {
  int N = pow(2,n);

  double *A, *cuda_A;
  Timer timer;

  // Allocate host memory and initialize
  A = (double*)malloc(N*N*sizeof(double));

  for (int i = 0; i < N*N; i++) {
    A[i] = i;
  }

  //print_A(A, N);


  // Allocate device memory and copy host data over
  CUDA_ERRCHK(cudaMalloc(&cuda_A, N*N*sizeof(double)));

  // copy data over
  CUDA_ERRCHK(cudaMemcpy(cuda_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice));

  // wait for previous operations to finish, then start timings
  CUDA_ERRCHK(cudaDeviceSynchronize());
  timer.reset();

  // Perform the transpose operation
  transpose<<<(N+255)/256, 256>>>(cuda_A, N);

  // wait for kernel to finish, then print elapsed time
  CUDA_ERRCHK(cudaDeviceSynchronize());
  double elapsed = timer.get();
  //std::cout << std::endl << "Time for transpose: " << elapsed << std::endl;
  //std::cout << "Effective bandwidth: " << (2*N*N*sizeof(double)) / elapsed * 1e-9 << " GB/sec" << std::endl;
  //std::cout << std::endl;
   bw = (2*N*N*sizeof(double)) / elapsed * 1e-9;
  if (N!=4096) std::cout << bw << ", ";

  // copy data back (implicit synchronization point)
  CUDA_ERRCHK(cudaMemcpy(A, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost));

  //print_A(A, N);

  CUDA_ERRCHK(cudaFree(cuda_A));
  free(A);

  CUDA_ERRCHK(cudaDeviceReset());  // for CUDA leak checker to work
}
  std::cout << bw << "]";

  return EXIT_SUCCESS;
}
