/// \file
/// \brief Widget class with affilicated functions (definitions)

#include "Widget.hpp"

/// \todo Add standard library headers if needed

namespace ex11 {

Widget value::init(Vector vec) {
  /// \todo improve implementation, if possible
    return Widget{std::move(vec)};
}

Widget lref::init(const Vector& vec) {
  /// \todo improve implementation, if possible
  return Widget{vec};
}

Widget rref::init(Vector&& vec) {
  /// \todo improve implementation, if possible
    return Widget{std::move(vec)};
}

} // namespace ex11
