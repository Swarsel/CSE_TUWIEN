#include <stdio.h>
#include "timer.hpp"

int main(void)
{

  int N[7] = {100, 300, 1000, 10000, 100000, 1000000};

  for(int j=0; j <= 6; j++){
    double *x, *d_x;
    Timer timer;
    double time = 0.0;
    double time_sum = 0.0;

    for(int it=0; it<10; it++){

      cudaDeviceSynchronize();
      timer.reset();
      x = (double*)malloc(N[j]*sizeof(double));
      cudaMalloc(&d_x, N[j]*sizeof(double));


      for (int i = 0; i < N[j]; i++) {
        x[i] = i * 1.0f;
        cudaMemcpy(&d_x[i], &x[i], sizeof(double), cudaMemcpyHostToDevice);
}
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
