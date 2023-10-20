#include <stdio.h>
#include "timer.hpp"

int main(void)
{

  int N[7] = {100, 300, 1000, 10000, 100000, 1000000, 3000000};

  for(int j=0; j <= 6; j++){
    double *x, *y, *d_x, *d_y;
    Timer timer;
    double time = 0.0;
    double time_sum = 0.0;

    for(int it=0; it<10; it++){

      cudaDeviceSynchronize();
      timer.reset();
      x = (double*)malloc(N[j]*sizeof(double));
      y = (double*)malloc(N[j]*sizeof(double));
      cudaMalloc(&d_x, N[j]*sizeof(double));
      cudaMalloc(&d_y, N[j]*sizeof(double));
      for (int i = 0; i < N[j]; i++) {
        x[i] = (double)i;
        y[i] = (double)(N[j]-i-1);
      }
      cudaMemcpy(d_x, x, N[j]*sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_y, y, N[j]*sizeof(double), cudaMemcpyHostToDevice);
      time = timer.get();

      cudaDeviceSynchronize();
      time_sum += time;
      free(x);
      cudaFree(d_x);
    }
    printf("%f\n", time_sum/10);
  }
  return EXIT_SUCCESS;
}
