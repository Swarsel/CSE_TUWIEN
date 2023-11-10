
__global__ void 1a(int N, double *x, double *result)
{
  __shared__ double sum[256]; // sum of all entries
  __shared__ double absum[256]; // sum of absolute value of all entries
  __shared__ double sqsum[256]; // sum of squares of all entries
  __shared__ double non0[256]; // number of zero entries

  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int total_threads = blockDim.x * gridDim.x;

  for (int i = tid; i < N; i += total_threads) {
    atomicAdd(&sum[threadIdx.x], x[i]);
    atomicAdd(&absum[threadIdx.x], x[i]);
    atomicAdd(&sqsum[threadIdx.x], x[i]);
    atomicAdd(&non0[threadIdx.x], x[i]);
  }

  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k /= 2) {
    if (threadIdx.x < k) {
      sum[threadIdx.x] += sum[threadIdx.x + k];
      absum[threadIdx.x] += absum[threadIdx.x + k];
      sqsum[threadIdx.x] += sqsum[threadIdx.x + k];
      non0[threadIdx.x] += non0[threadIdx.x + k];
      __syncthreads();
      }
  }

  if (threadIdx.x == 0) {
    atomicAdd(&out[0], sum[0]);
    atomicAdd(&out[1], absum[0]);
    atomicAdd(&out[2], sqsum[0]);
    atomicAdd(&out[3], non0[0]);
  }
}
