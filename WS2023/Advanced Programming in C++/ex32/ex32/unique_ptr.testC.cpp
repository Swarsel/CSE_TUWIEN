/// \file
/// \brief Testing ex32::unique_ptr

#include "unique_ptr.hpp" // ex32::unique_ptr

#include <cassert> // assert

int main() {

  using namespace ex32;

  struct Widget {
    double m = 42;
  };

  using T = Widget;

  { // compare to same type
    auto up1 = unique_ptr(new T{});
    auto up2 = unique_ptr(new T{});
    assert(up1 != up2);               // expect comparison based on the managed resource
    auto up3 = unique_ptr(up2.get()); // note: this requires the release() below
    assert(up2 == up3);               // expect comparison based on the managed resource
    up3.release();                    // release the resource (avoid a double free/delete)
  }

  { // compare to nullptr
    auto up = unique_ptr(new T{});
    assert(up !=
           nullptr); //  expect a comparison with nullptr is allowed and reflects the current state
    up = nullptr;    // expect assigning a nullptr works
    assert(up == nullptr); //  expect assignment of nullptr has effect on managed resource
  }

  { // use as bool
    auto up = unique_ptr(new T{});
    [[maybe_unused]] bool isnull = up;
    assert(isnull == false); // expect implicit conversion to bool to reflect current state
  }
  { // use as bool
    auto up = unique_ptr<T>(nullptr); 
    [[maybe_unused]] bool isnull = up;
    assert(isnull == true); // expect implicit conversion to bool to reflect current state
  }

  return 0;
}
