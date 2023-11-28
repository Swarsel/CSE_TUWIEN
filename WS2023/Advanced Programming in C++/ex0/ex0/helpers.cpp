/// \file
/// \brief Definitions of helper functions for std::vector<double>

#include "helpers.hpp"
#include <cstddef>    // std::size_t
#include <functional> // std::function
#include <vector>     // std::vector
#include <cstdio>
#include <algorithm>
#include <random>

/// \todo Add standard library headers as needed

namespace ex0 {

/// \todo Implement the seven functions below

    void print(const Vector& vec){
        for (const auto& item: vec) {
            std::printf("%f\n", item);
        }
    }
    void reset(Vector& vec) {
        vec.clear();
            }
    Vector copy(const Vector& vec) {
        Vector new_vec;
        new_vec = vec;
        return new_vec;
    }
    Vector concat(const Vector& a, const Vector& b) {
        Vector v;
        v.insert(v.end(), a.begin(), a.end());
        v.insert(v.end(), b.begin(), b.end());
        return v;
    }
    void swap(Vector& a, Vector& b) {
        a.swap(b);
    }
    void fill_uniform_random(Vector& vec, std::size_t n, double lower, double upper) {
        std::random_device rd;
        std::mt19937 gen(rd());
        vec.resize(n);
        std::uniform_int_distribution<> dis(lower, upper);
        std::generate(vec.begin(), vec.end(), [&](){ return dis(gen); });
    }
    void sort(Vector& vec, Compare comp) {
        std::sort(vec.begin(), vec.end(), comp);
    }
} // namespace ex0
