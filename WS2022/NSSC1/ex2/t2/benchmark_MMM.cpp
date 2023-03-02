#include <assert.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// include for OpenBLAS
#include <openblas/cblas.h>
// include for Eigen 
#include <eigen3/Eigen/Dense>
#include <chrono>

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
  
  int elem;
  for (size_t i = 0; i<N; i++) {
    for (size_t j = j; j<N; j++) {
      elem = i+1 + (j+1)*N;
      matrix1[i*N+j] = elem;
      matrix2[j*N+i] = elem;
    }
  }

  Eigen::Map<Eigen::MatrixXd> m1(matrix1.data(),N, N);
  Eigen::Map<Eigen::MatrixXd> m2(matrix2.data(),N, N);
  Eigen::Map<Eigen::MatrixXd> m3(matrixResult.data(),N, N);

  std::chrono::steady_clock::time_point t0;
  std::chrono::steady_clock::time_point t1;
  
  if (impl == "CUSTOM") {
    t0= std::chrono::steady_clock::now();
    MMMCustom(matrix1, matrix2, matrixResult, N);
    t1 = std::chrono::steady_clock::now();
  } else if (impl == "OPENBLAS") {
  //https://www.intel.com/content/www/us/en/develop/documentation/mkl-tutorial-c/top/multiplying-matrices-using-dgemm.html
    t0= std::chrono::steady_clock::now();
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N,
                1.0, matrix1.data(), N, matrix2.data(), N, 0.0, matrixResult.data(), N);
    t1 = std::chrono::steady_clock::now();            
  } else {
    t0= std::chrono::steady_clock::now();
    m3 = m1 * m2;
    t1 = std::chrono::steady_clock::now();
  }
 
  std::chrono::duration<double> result = std::chrono::duration_cast<std::chrono::duration<double>>(t1-t0);

  bool sym = true;
  if ((impl == "CUSTOM") || (impl == "OPENBLAS")){
    for (size_t i = 0; i<N; i++) {
      for (size_t j = i; j<N; j++) {
        if (matrixResult[i*N+j] != matrixResult[j*N+i]) sym = false;
      }
    }
  }
  else {
    for (size_t i = 0; i<N; i++) {
      for (size_t j = i; j<N; j++) {
        if (m3(i,j) != m3(j,i)) sym = false;
      }
    }
  }
  
  
 std::cout << "Symmetric: " << sym  <<", time: " << result.count() << "s." << std::endl;
  
  
  return 0;
}
