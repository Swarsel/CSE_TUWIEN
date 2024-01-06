/// \file
/// \brief Testing ex32::unique_ptr

#include "unique_ptr.hpp" // ex32::unique_ptr

#include <cassert>     // assert

int main() {

  using namespace ex32;

  struct Widget {
    double m = 42;
  };

  using T = Widget;

  { // reset(arg): should delete 'old' managed resource and start to manage arg
    auto up = unique_ptr(new T{});
    auto* ptr = new T{};
    up.reset(ptr);
    assert(up.get() == ptr); // expect reset(ptr) to change managed resource to ptr
    up.reset();
    assert(up.get() == nullptr); // expect reset() to end management of resource
  }

  { // release(): should release (not delete) managed resource
    auto* ptr1 = new T{};
    {
      auto up = unique_ptr(ptr1);
      [[maybe_unused]] auto* ptr2 = up.release();
      assert(ptr1 == ptr2);        // expect release() to return managed resource
      assert(up.get() == nullptr); // expect release() to end management of resource
    }
    // need to delete original ressource because it was released from unique_ptr
    delete ptr1;
  }

  { // assign nullptr
    auto up = unique_ptr(new T{});
    up = nullptr;
    assert(up.get() == nullptr); // expect assignment from nullptr to release resource
  }

  return 0;
}
