#ifndef PERLIN_NOISE_H
#define PERLIN_NOISE_H

#include "noise_base.h"
#include <array>

class PerlinNoise : public NoiseBase {
public:
    PerlinNoise(unsigned int seed = 0);
    
    float evaluate2D(float x, float y) override;
    std::string getName() const override { return "Perlin Noise"; }
    
private:
    std::array<int, 512> permutation;
    std::array<float, 512> gradients1D;
    std::array<NoiseUtils::Vec2, 256> gradients2D;
    
    void initializePermutation(unsigned int seed);
    void initializeGradients();
    
    float fade(float t);
    int hash(int x, int y);
    float grad(int hash, float x, float y);
};

#endif // PERLIN_NOISE_H