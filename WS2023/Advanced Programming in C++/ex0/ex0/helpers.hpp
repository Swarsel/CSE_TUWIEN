/// \file
/// \brief Declarations of helper functions for std::vector<double>

#include <cstddef>    // std::size_t
#include <functional> // std::function
#include <vector>     // std::vector

namespace ex0 {


/// \brief Type alias for the std::vector with the type of elements fixed to double
using Vector = std::vector<double>;

/// \brief Prints all elements in a Vector to the console
/// \param vec The Vector to operate on
void print(const Vector& vec);

/// \brief Removes all elements from a Vector
/// \param vec The Vector to operate on
void reset(Vector& vec);

/// \brief Produces a copy of a Vector
/// \param vec The Vector to be copied
/// \return A copy of vec
Vector copy(const Vector& vec);

/// \brief Concatenates the elements of two Vectors
/// \param a First Vector
/// \param b Second Vector
/// \return Vector containing the elements of a and b
Vector concat(const Vector& a, const Vector& b);

/// \brief Swaps the elements of two Vectors
/// \param a First Vector
/// \param b Second Vector
void swap(Vector& a, Vector& b);

/// \brief Fills a Vector with random numbers uniformly distributed in a given interval
/// \param vec The Vector to operate on
/// \param n Number of random values
/// \param lower Lower bound of the interval
/// \param upper Upper bound of the interval
void fill_uniform_random(Vector& vec, std::size_t n, double lower, double upper);

/// \brief Type used for the comparions function in sort
using Compare = std::function<bool(const double& a, const double& b)>;

/// \brief Sorts a Vector according to a comparison function
/// \param vec The Vector to be sorted
/// \param comp Comparison function which returns true if the first argument occurs before the
/// second argument in a sorted sequence
void sort(Vector& vec, Compare comp);

} // namespace ex0