/// \file
/// \brief Benchmarking the wrapper functions by adding large elements to a list

#include <ex21/Wrappers.hpp>

#include <chrono>   // std::chrono*
#include <iostream> // std::cout|endl
#include <utility>  // std::forward

#include <list>   // std::list
#include <vector> // std::vector
 
int main() {

  std::size_t n = 10'000'000; // length for test vector

  using Vector = std::vector<double>;
  using List = std::list<Vector>;

  using namespace ex21;

  {
    std::cout << "push_back(const Vector&)            : ";
    List lst;
    auto vec = Vector(n, 42);

    auto push_back = [&lst](auto&&... args) -> auto {
      lst.push_back(std::forward<decltype(args)>(args)...);
    };

    timeit(push_back, vec);
  }

  {
    std::cout << "emplace_back(/*ctor args*/);        : ";
    List lst;
    auto vec = Vector(n, 42);

    auto emplace_back = [&lst](auto&&... args) -> auto {
      lst.emplace_back(std::forward<decltype(args)>(args)...);
    };

    timeit(emplace_back, n, 42);
  }

  {
    std::cout << "push_back(Vector&&); from std::move : ";
    List lst;
    auto vec = Vector(n, 42);

    auto push_back = [&lst](auto&&... args) -> auto {
      lst.push_back(std::forward<decltype(args)>(args)...);
    };

    timeit(push_back, std::move(vec));
  }

  {
    std::cout << "push_back(Vector&&); from temporary : ";
    List lst;
    auto vec = Vector(n, 42);

    auto push_back = [&lst](auto&&... args) -> auto {
      lst.push_back(std::forward<decltype(args)>(args)...);
    };

    timeit(push_back, Vector(n, 42));
  }

  return 0;
}
