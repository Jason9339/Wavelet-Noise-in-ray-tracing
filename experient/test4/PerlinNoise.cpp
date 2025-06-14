#include "PerlinNoise.hpp"
#include "CommonUtils.hpp" // For fade, lerp
#include <numeric>        // For std::iota
#include <algorithm>      // For std::shuffle
#include <cmath>          // For std::floor

PerlinNoise::PerlinNoise(unsigned int seed) {
    p.resize(256);
    std::iota(p.begin(), p.end(), 0); // Fill with 0..255

    std::mt19937 g(seed); // Use a specific seed for reproducibility if needed
    std::shuffle(p.begin(), p.end(), g);

    p.insert(p.end(), p.begin(), p.end()); // Duplicate to avoid modulo 256

    // Precompute some 2D gradients
    // Using 4 diagonal unit vectors
    gradients.push_back(Vec2(1,1).normalized());
    gradients.push_back(Vec2(-1,1).normalized());
    gradients.push_back(Vec2(1,-1).normalized());
    gradients.push_back(Vec2(-1,-1).normalized());
    gradients.push_back(Vec2(1,0));
    gradients.push_back(Vec2(-1,0));
    gradients.push_back(Vec2(0,1));
    gradients.push_back(Vec2(0,-1));
}

// Simple hash for 2D. Could be improved but okay for visualization.
int PerlinNoise::hash(int x, int y) const {
    // Ensure positive before modulo for safety with some % implementations
    int X = x & 255;
    int Y = y & 255;
    return p[p[X] + Y]; 
}


Vec2 PerlinNoise::getGradient(int ix, int iy) const {
    int h = hash(ix, iy);
    return gradients[h % gradients.size()];
}


float PerlinNoise::noise(float x, float y) const {
    int ix0 = static_cast<int>(std::floor(x));
    int iy0 = static_cast<int>(std::floor(y));
    int ix1 = ix0 + 1;
    int iy1 = iy0 + 1;

    float fx = x - static_cast<float>(ix0);
    float fy = y - static_cast<float>(iy0);

    float u = fade(fx);
    float v = fade(fy);

    // Get gradients at the 4 corners
    Vec2 grad00 = getGradient(ix0, iy0);
    Vec2 grad10 = getGradient(ix1, iy0);
    Vec2 grad01 = getGradient(ix0, iy1);
    Vec2 grad11 = getGradient(ix1, iy1);

    // Compute dot products
    // Verification output for one point (e.g., near origin)
    // if (std::abs(x) < 0.1 && std::abs(y) < 0.1) {
    //    std::cout << "Perlin Debug: (" << x << "," << y << ")\n";
    //    std::cout << "  ix0=" << ix0 << ", iy0=" << iy0 << ", fx=" << fx << ", fy=" << fy << "\n";
    //    std::cout << "  grad00=(" << grad00.x << "," << grad00.y << ")\n";
    // }

    float dot00 = grad00.dot(Vec2(fx, fy));
    float dot10 = grad10.dot(Vec2(fx - 1.0f, fy));
    float dot01 = grad01.dot(Vec2(fx, fy - 1.0f));
    float dot11 = grad11.dot(Vec2(fx - 1.0f, fy - 1.0f));
    
    // Interpolate
    float val_x0 = lerp(dot00, dot10, u);
    float val_x1 = lerp(dot01, dot11, u);
    float result = lerp(val_x0, val_x1, v);

    // Result is roughly in [-sqrt(2)/2, sqrt(2)/2] or [-1, 1] if gradients are not normalized
    // and up to 4 unit vectors.
    // For normalized gradients like (1,1)/sqrt(2), max dot is 1. Typical range is often taken as [-1, 1].
    // Let's assume it's roughly in [-0.7, 0.7] to [-1,1] depending on gradients.
    return result; 
}