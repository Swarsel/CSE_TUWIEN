#include <stdio.h>
#include <math.h>
#include <iostream>
#include "timer.hpp"
#include "cuda_errchk.hpp"   // for error checking of CUDA calls

__global__ void transpose(double *A)
{
  // the +1's are for avoiding shared memory bank conflicts
  __shared__ float tile_1[16+1][16+1];
  __shared__ float tile_2[16+1][16+1];

  int x = blockIdx.x * 16 + threadIdx.x;
  int y = blockIdx.y * 16 + threadIdx.y;
  int width = gridDim.x * 16;
  int t_x = blockIdx.y * 16 + threadIdx.x;
  int t_y = blockIdx.x * 16 + threadIdx.y;

  tile_1[threadIdx.y][threadIdx.x] = A[y * width + x];
  tile_2[threadIdx.y][threadIdx.x] = A[t_y * width + t_x];

  __syncthreads();

  // for diagonal elements just perform the transpose, for off-diagonal elements also swap the respective tiles
  if (blockIdx.y < blockIdx.x) A[t_y * width + t_x] = tile_1[threadIdx.x][threadIdx.y];
  A[y * width + x] = tile_2[threadIdx.x][threadIdx.y];
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

  dim3 dimGrid(N/16, N/16, 1);
  dim3 dimBlock(16, 8, 1);

  double *A, *cuda_A;

  Timer timer;

  // Allocate host memory and initialize
  A = (double*)malloc(N*N*sizeof(double));

  for (int i = 0; i < N*N; i++) {
    A[i] = i;
  }

  // print_A(A, N);


  // Allocate device memory and copy host data over
  CUDA_ERRCHK(cudaMalloc(&cuda_A, N*N*sizeof(double)));

  // copy data over
  CUDA_ERRCHK(cudaMemcpy(cuda_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice));

  // wait for previous operations to finish, then start timings
  CUDA_ERRCHK(cudaDeviceSynchronize());

  timer.reset();

  // Perform the transpose operation
  transpose<<<dimGrid, dimBlock>>>(cuda_A);

  // wait for kernel to finish, then print elapsed time
  CUDA_ERRCHK(cudaDeviceSynchronize());
  double elapsed = timer.get();

  //std::cout << std::endl << "Time for transpose: " << elapsed << std::endl;
  //std::cout << "Effective bandwidth: " << (2*N*N*sizeof(double)) / elapsed * 1e-9 << " GB/sec" << std::endl;
  bw = (2*N*N*sizeof(double)) / elapsed * 1e-9;
  //std::cout << N << ", " << elapsed << ", " << bw << std::endl;

  if (N!=4096) std::cout << bw << ", ";

  // copy data back (implicit synchronization point)
  CUDA_ERRCHK(cudaMemcpy(A, cuda_A, N*N*sizeof(double), cudaMemcpyDeviceToHost));

  // print_A(A, N);

  CUDA_ERRCHK(cudaFree(cuda_A));
  free(A);

  CUDA_ERRCHK(cudaDeviceReset());  // for CUDA leak checker to work
  }
  std::cout << bw << "]";
  return EXIT_SUCCESS;
}
