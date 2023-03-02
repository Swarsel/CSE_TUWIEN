// compile using: clang++ -std=c++17 -O3 -march=native benchmark_triad.cpp
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

using Clock = std::chrono::steady_clock;
using Duration = std::chrono::duration<double>;
using Vector = std::vector<double>;

void triad(Vector &a, const Vector &b, const Vector &c, const Vector &d) {
  auto aptr = a.data();
  const auto bptr = b.data();
  const auto cptr = c.data();
  const auto dptr = d.data();
//  #pragma clang loop vectorize(enable)
  for (int i = 0; i < c.size(); ++i)
    aptr[i] = bptr[i] + cptr[i] * dptr[i];
}

template <typename... ARGS> auto measure(ARGS &&...args) {
  auto start = Clock::now();
  triad(std::forward<ARGS>(args)...);
  auto stop = Clock::now();
  return Duration(stop - start).count();
}

struct Test {
  int N; // vector length 
  double footprint; // memory footprint in kbytes
  double flops; // total number of flops to be performed
  double runtime = 0; // runtime per sweep
  Test(int N) : N(N), footprint(N * 4.0 * 8.0 / 1024.0), flops(N * 2) {}
};

int main() {
  std::vector<Test> Tests = {200,       300,      400,     800,     2'000,
                             4'000,     6'000,    8'000,   20'000,  40'000,
                             60'000,    100'000,  200'000, 300'000, 1'000'000,
                             5'000'000, 2'000'000, 10'000'000};

  for (auto &item : Tests) {
    Vector a(item.N, 1.0), b(item.N, 1.0), c(item.N, 1.0), d(item.N, 1.0);
    for (int n = 0; n != 19; ++n)
      measure(a, b, c, d); // warmup
    int N = 40;
    for (int n = 0; n != N; ++n)
      item.runtime += measure(a, b, c, d);
    item.runtime /= N;
  }

  for (auto &item : Tests) {
    std::cout << item.N << ", " << item.footprint << ", " << item.flops << ", "
              << item.runtime << ", " << item.flops / item.runtime << std::endl;
  }

  return 0;
}
