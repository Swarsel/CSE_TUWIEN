#include "timer.hpp"
#include <iostream>

#define VIENNACL_WITH_CUDA

#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/vector.hpp"

int main(int argc, char *argv[]) {

  Timer timer;
  int N = std::atoi(argv[1]);
  double time = 0;
  double result = 0;
  viennacl::vector<double> x = viennacl::scalar_vector<double>(N, 1.0);
  viennacl::vector<double> y = viennacl::scalar_vector<double>(N, 2.0);

  timer.reset();

  result = viennacl::linalg::inner_prod(x + y, x - y);

  time = timer.get();

  std::cout << time << std::endl;

  return EXIT_SUCCESS;
}
