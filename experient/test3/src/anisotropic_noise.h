#ifndef ANISOTROPIC_NOISE_H
#define ANISOTROPIC_NOISE_H

#include "noise_base.h"
#include <vector>
#include <complex>

class AnisotropicNoise : public NoiseBase {
public:
    AnisotropicNoise(int numOrientations = 4, float anisotropy = 0.5f);
    
    float evaluate2D(float x, float y) override;
    std::string getName() const override { return "Anisotropic Noise"; }
    
private:
    int numOrientations;
    float anisotropy;
    std::vector<std::vector<std::vector<float>>> orientedBands;
    
    void generateOrientedBands();
    std::vector<std::vector<float>> generateSteerableFilter(float angle, int size);
    std::vector<std::vector<float>> applyFilter(const std::vector<std::vector<float>>& noise,
                                               const std::vector<std::vector<float>>& filter);
};

#endif // ANISOTROPIC_NOISE_H