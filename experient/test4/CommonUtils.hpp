#ifndef COMMON_UTILS_HPP
#define COMMON_UTILS_HPP

#include <random>
#include <vector>
#include <algorithm> // For std::shuffle
#include <numeric>   // For std::iota

// Standard Mersenne Twister engine
static std::mt19937 rng(std::random_device{}());

inline float random_float(float min, float max) {
    std::uniform_real_distribution<float> dist(min, max);
    return dist(rng);
}

// Quintic smoothstep
inline float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

inline float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

// Quadratic B-spline basis function B(t) for t in [0, 3)
// This is for evaluating N(x) = sum n_i B(2x-i) as per CD05
// B_quad(t) = 0.5*t^2              if 0 <= t < 1
//             -t^2 + 3t - 1.5      if 1 <= t < 2
//             0.5*(3-t)^2          if 2 <= t < 3
//             0                    otherwise
inline float b_spline_quadratic_eval(float t) {
    if (t < 0.0f || t >= 3.0f) return 0.0f;
    if (t < 1.0f) return 0.5f * t * t;
    if (t < 2.0f) return -t * t + 3.0f * t - 1.5f;
    // t < 3.0f
    float term = 3.0f - t;
    return 0.5f * term * term;
}

#endif // COMMON_UTILS_HPP