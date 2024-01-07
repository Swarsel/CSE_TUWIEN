#include "timer.hpp"
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <vector>

int main(int argc, char *argv[]) {

  int N = std::atoi(argv[1]);
  double time = 0;
  Timer timer;

  // generate 32M random numbers on the host
  thrust::device_vector<float> x(N, 1.0);
  thrust::device_vector<float> y(N, 2.0);
  thrust::device_vector<float> plus(N);
  thrust::device_vector<float> minus(N);

  double result = 0;
  timer.reset();
  thrust::transform(x.begin(), x.end(), y.begin(), plus.begin(),
                    thrust::plus<double>());
  thrust::transform(x.begin(), x.end(), y.begin(), minus.begin(),
                    thrust::minus<double>());

  result = thrust::inner_product(plus.begin(), plus.end(), minus.begin(), 0.0f);

  time = timer.get();

  std::cout << time << std::endl;

  return 0;
}
