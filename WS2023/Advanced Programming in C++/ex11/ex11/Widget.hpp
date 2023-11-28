/// \file
/// \brief Widget class with affilicated functions (declarations)

#pragma once

#include <vector> // std::vector

namespace ex11 {

/// \brief Type alias for the std::vector with the type of elements fixed to double
using Vector = std::vector<double>;

/// \brief Simple wrapper class holding a Vector
struct Widget {
  Vector vec;
};

namespace value {
/// \brief Constructs a Widget from an by-value reference parameter
/// \return A Widget constructed from the argument
Widget init(Vector vec);
} // namespace value

namespace lref {
/// \brief Constructs a Widget from an l-value reference parameter
/// \return A Widget constructed from the argument
Widget init(const Vector& vec);
} // namespace lref

namespace rref {
/// \brief Constructs a Widget from an r-value reference parameter
/// \return A Widget constructed from the argument
Widget init(Vector&& rref);
} // namespace rref

} // namespace ex11
