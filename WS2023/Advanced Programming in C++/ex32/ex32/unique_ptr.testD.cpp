/// \file
/// \brief Testing ex32::unique_ptr

#include "unique_ptr.hpp" // ex32::unique_ptr

#include <cassert> // assert
#include <cstdio>  // std::fopen|fprintf|fclose

// globals
int delete_double_calls = 0;
int delete_double2_calls = 0;
int delete_array_calls = 0;
int delete_FILE_calls = 0;

void deleter_double(double* p) {
  delete p;
  ++delete_double_calls;
}

void deleter_double2(double* p) {
  delete p;
  ++delete_double2_calls;
}

void deleter_FILE(FILE* handle) {
  std::fclose(handle);
  ++delete_FILE_calls;
}

void deleter_array_double(double* p) {
  delete[] p;
  ++delete_array_calls;
}

auto open(const char* name, const char* mode) {}

int main() {

  using namespace ex32;

  { // custom deleter double
    auto up1 = unique_ptr(new double{}, deleter_double);
  }
  assert(delete_double_calls == 1); // expect custom deleter to be used (once)

  { // custom deleter for a raw array
    auto up = unique_ptr(new double[10], deleter_array_double);
  }
  assert(delete_array_calls == 1); // expect custom deleter to be used (once)

  { // custom "closer" as deleter
    auto filename = "data.json";
    auto mode = "w+";
    FILE* ptr = std::fopen(filename, mode);
    auto file = unique_ptr(ptr, deleter_FILE);
    std::fprintf(file.get(), "%s", "{\n  \"unique_ptr.testD.cpp\": \"success\"\n}\n");
  }
  assert(delete_FILE_calls == 1); // expect custom deleter to be used (once)

  { // custom deleter should be moved when assigning
    auto up1 = unique_ptr(new double{}, deleter_double2);
    {
      auto up2 = unique_ptr(new double{}, deleter_double);
      up2 = std::move(up1); // expect custom deleter to be moved, too
    }
    assert(delete_double2_calls == 1); // expect custom deleter to be used (once)
  }
  assert(delete_double_calls == 2); // expect custom deleter to be used (once)

  return 0;
}
