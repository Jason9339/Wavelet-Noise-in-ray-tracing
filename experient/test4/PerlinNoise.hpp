#ifndef PERLIN_NOISE_HPP
#define PERLIN_NOISE_HPP

#include "Vec2.hpp"
#include <vector>
#include <iostream> // For verification output

class PerlinNoise {
public:
    PerlinNoise(unsigned int seed = 12345);
    float noise(float x, float y) const;

private:
    std::vector<int> p; // Permutation table
    std::vector<Vec2> gradients; // Precomputed gradient vectors

    int hash(int x, int y) const;
    Vec2 getGradient(int ix, int iy) const;
};

#endif // PERLIN_NOISE_HPP