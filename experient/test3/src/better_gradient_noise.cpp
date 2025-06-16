#include "better_gradient_noise.h"
#include <cmath>
#include <algorithm>

BetterGradientNoise::BetterGradientNoise(unsigned int seed) {
    NoiseUtils::log("BetterGradientNoise: Initializing with seed " + std::to_string(seed));
    initializePermutation(seed);
    initializeGradients();
}

void BetterGradientNoise::initializePermutation(unsigned int seed) {
    NoiseUtils::log("BetterGradientNoise: Initializing improved permutation table");
    
    // Initialize with values 0-255
    for (int i = 0; i < 256; i++) {
        permutation[i] = i;
    }
    
    // Better shuffling algorithm
    std::mt19937 gen(seed);
    for (int i = 255; i > 0; i--) {
        std::uniform_int_distribution<int> dis(0, i);
        int j = dis(gen);
        std::swap(permutation[i], permutation[j]);
    }
    
    // Duplicate for overflow
    for (int i = 0; i < 256; i++) {
        permutation[256 + i] = permutation[i];
    }
}

void BetterGradientNoise::initializeGradients() {
    NoiseUtils::log("BetterGradientNoise: Initializing improved gradient table");
    
    // Use more evenly distributed gradients
    for (int i = 0; i < 256; i++) {
        float angle = 2.0f * M_PI * i / 256.0f;
        gradients[i] = NoiseUtils::Vec2(cos(angle), sin(angle));
    }
}

float BetterGradientNoise::smootherstep(float t) {
    // Improved smoothing function
    return t * t * t * (t * (t * 6 - 15) + 10);
}

int BetterGradientNoise::hash(int x, int y) {
    // Improved hash function to reduce axial bias
    const int prime1 = 73856093;
    const int prime2 = 19349663;
    int n = x * prime1 ^ y * prime2;
    n = n ^ (n >> 16);
    return permutation[n & 255];
}

float BetterGradientNoise::grad(int hash, float x, float y) {
    const NoiseUtils::Vec2& g = gradients[hash & 255];
    return g.x * x + g.y * y;
}

float BetterGradientNoise::evaluate2D(float x, float y) {
    // Find unit square
    int X = static_cast<int>(std::floor(x));
    int Y = static_cast<int>(std::floor(y));
    
    // Relative position in square
    float fx = x - X;
    float fy = y - Y;
    
    // Smootherstep interpolation
    float u = smootherstep(fx);
    float v = smootherstep(fy);
    
    // Hash coordinates of square corners
    int h00 = hash(X, Y);
    int h10 = hash(X + 1, Y);
    int h01 = hash(X, Y + 1);
    int h11 = hash(X + 1, Y + 1);
    
    // Compute gradients
    float g00 = grad(h00, fx, fy);
    float g10 = grad(h10, fx - 1, fy);
    float g01 = grad(h01, fx, fy - 1);
    float g11 = grad(h11, fx - 1, fy - 1);
    
    // Interpolate
    float x0 = NoiseUtils::lerp(g00, g10, u);
    float x1 = NoiseUtils::lerp(g01, g11, u);
    
    return NoiseUtils::lerp(x0, x1, v);
}