#ifndef BETTER_GRADIENT_NOISE_H
#define BETTER_GRADIENT_NOISE_H

#include "noise_base.h"
#include <array>

class BetterGradientNoise : public NoiseBase {
public:
    BetterGradientNoise(unsigned int seed = 0);
    
    float evaluate2D(float x, float y) override;
    std::string getName() const override { return "Better Gradient Noise"; }
    
private:
    std::array<int, 512> permutation;
    std::array<NoiseUtils::Vec2, 256> gradients;
    
    void initializePermutation(unsigned int seed);
    void initializeGradients();
    
    float smootherstep(float t);
    int hash(int x, int y);
    float grad(int hash, float x, float y);
};

#endif // BETTER_GRADIENT_NOISE_H