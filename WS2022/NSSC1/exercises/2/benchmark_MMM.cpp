#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// include for OpenBLAS
// #include <cblas.h>
// include for Eigen 
// #include <eigen3/Eigen/Dense>

void MMMCustom(const std::vector<double> &m1, const std::vector<double> &m2,
               std::vector<double> &res, const size_t N) {
  assert(m1.size() == N * N && m2.size() == N * N && res.size() == N * N);
  for (size_t j = 0; j != N; ++j) {
    for (size_t i = 0; i != N; ++i) {
      double tmp = 0;
      for (size_t k = 0; k != N; ++k) {
        tmp += m1[k + i * N] * m2[j + k * N];
      }
      res[j + i * N] += tmp;
    }
  }
}

int main(int argc, char *argv[]) {

  // parse command line arguments
  assert(argc == 3);
  std::string impl = "UNKNOWN";
  int size = -1;
  {
    std::istringstream tmp(argv[1]);
    tmp >> impl;
  }
  {
    std::istringstream tmp(argv[2]);
    tmp >> size;
  }
  std::cout << "impl=" << impl << std::endl;
  std::cout << "size=" << size << std::endl;
  assert(size > 0);
  assert(impl == "CUSTOM");
  size_t N = size;

  // allocate matrices
  std::vector<double> matrix1(N * N, 0);
  std::vector<double> matrix2(N * N, 0);
  std::vector<double> matrixResult(N * N, 0);

  MMMCustom(matrix1, matrix2, matrixResult, N);

  return 0;
}