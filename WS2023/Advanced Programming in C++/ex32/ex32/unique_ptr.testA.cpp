/// \file
/// \brief Testing ex32::unique_ptr

#include "unique_ptr.hpp" // ex32::unique_ptr

#include <cassert>     // assert
#include <type_traits> // std::is_same|is_copy_constructible|is_copy_assignable
#include <utility>     // std::declval|move

int main() {

  using namespace ex32;

  struct Widget {
    double m = 42;
  };

  using T = Widget;
  using UP = unique_ptr<T>;

  { // nested types
    using expected_pointer_type = T*;
    using expected_element_type = T;
    static_assert(std::is_same<UP::pointer, expected_pointer_type>::
                      value); // expected nested 'pointer' type to reflect the manager pointer T
    static_assert(std::is_same<UP::element_type, expected_element_type>::value);
    // expected nested 'element_type' type to reflect the type of the managed object
  }

  { // .get() member
    static_assert(std::is_same<decltype(std::declval<UP>().get()),
                               UP::pointer>::value); // expect get() to return pointer to
                                                     // managed type
  }

  { // constructor from raw pointer
    const auto ptr1 = new T{};
    auto up1 = unique_ptr(ptr1); // expect to be constructable from raw pointer
    [[maybe_unused]] const auto ptr2 = up1.get();
    assert(ptr1 == ptr2); // expect to manage raw pointer used for construction
  }

  { // disabled copy semantics

    static_assert(!std::is_copy_constructible<UP>::value); //  expect NOT to be copy constructible
    static_assert(!std::is_copy_assignable<UP>::value);    //  expect NOT to be copy assignable
  }

  { // move constructor
    auto up1 = unique_ptr(new T{});
    [[maybe_unused]]  const auto ptr1 = up1.get();
    auto up2 = unique_ptr(std::move(up1)); // expect to be move constructible
    assert(up1.get() == nullptr);          // expect moved-from instance to NOT manage any resource
    assert(ptr1 == up2.get()); // expect moved constructed instance to manage the original resource
  }

  { // move assignment
    auto up1 = unique_ptr(new T{});
    auto up2 = unique_ptr(new T{});
    [[maybe_unused]] const auto ptr1 = up1.get();
    [[maybe_unused]] const auto ptr2 = up2.get();

    up2 = std::move(up1); // expect ability to move-assign

    assert(up2.get() == ptr1); // expect move-assigned instance to manage the original resource
    assert(up1.get() == nullptr ||
           up1.get() == ptr2); // expect moved-from object to manage none or the other instance
  }

  return 0;
}
