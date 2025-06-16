#include "perlin_noise.h"
#include <cmath>
#include <algorithm>

PerlinNoise::PerlinNoise(unsigned int seed) {
    NoiseUtils::log("PerlinNoise: Initializing with seed " + std::to_string(seed));
    initializePermutation(seed);
    initializeGradients();
}

void PerlinNoise::initializePermutation(unsigned int seed) {
    NoiseUtils::log("PerlinNoise: Initializing permutation table");
    
    // Initialize with values 0-255
    for (int i = 0; i < 256; i++) {
        permutation[i] = i;
    }
    
    // Shuffle using seed
    std::mt19937 gen(seed);
    std::shuffle(permutation.begin(), permutation.begin() + 256, gen);
    
    // Duplicate for overflow
    for (int i = 0; i < 256; i++) {
        permutation[256 + i] = permutation[i];
    }
}

void PerlinNoise::initializeGradients() {
    NoiseUtils::log("PerlinNoise: Initializing gradient tables");
    
    // 2D gradients - use 12 directions like in the paper
    const float angles[] = {0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330};
    
    for (int i = 0; i < 12; i++) {
        float angle = angles[i] * M_PI / 180.0f;
        gradients2D[i] = NoiseUtils::Vec2(cos(angle), sin(angle));
    }
    
    // Repeat to fill array
    for (int i = 12; i < 256; i++) {
        gradients2D[i] = gradients2D[i % 12];
    }
}

float PerlinNoise::fade(float t) {
    // 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10);
}

int PerlinNoise::hash(int x, int y) {
    return permutation[permutation[x & 255] + (y & 255)];
}

float PerlinNoise::grad(int hash, float x, float y) {
    NoiseUtils::Vec2& g = gradients2D[hash & 255];
    return g.x * x + g.y * y;
}

float PerlinNoise::evaluate2D(float x, float y) {
    // Find unit square
    int X = static_cast<int>(std::floor(x)) & 255;
    int Y = static_cast<int>(std::floor(y)) & 255;
    
    // Relative position in square
    x -= std::floor(x);
    y -= std::floor(y);
    
    // Fade curves
    float u = fade(x);
    float v = fade(y);
    
    // Hash coordinates of square corners
    int a = hash(X, Y);
    int b = hash(X + 1, Y);
    int c = hash(X, Y + 1);
    int d = hash(X + 1, Y + 1);
    
    // Blend results from corners
    float x1 = NoiseUtils::lerp(grad(a, x, y), grad(b, x - 1, y), u);
    float x2 = NoiseUtils::lerp(grad(c, x, y - 1), grad(d, x - 1, y - 1), u);
    
    return NoiseUtils::lerp(x1, x2, v);
}