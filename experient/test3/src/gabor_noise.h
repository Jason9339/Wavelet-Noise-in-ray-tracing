#ifndef GABOR_NOISE_H
#define GABOR_NOISE_H

#include "noise_base.h"
#include <vector>

class GaborNoise : public NoiseBase {
public:
    GaborNoise(float kernelRadius = 0.1f, float frequency = 10.0f, 
               float orientation = 0.0f, float bandwidth = 1.0f);
    
    float evaluate2D(float x, float y) override;
    std::string getName() const override { return "Gabor Noise"; }
    
private:
    float kernelRadius;
    float frequency;
    float orientation;
    float bandwidth;
    
    struct GaborKernel {
        float x, y;
        float weight;
        float phase;
        float orientation;
    };
    
    std::vector<GaborKernel> generateKernels(int cellX, int cellY);
    float evaluateKernel(float x, float y, const GaborKernel& kernel);
    unsigned int hash(int x, int y);
};

#endif // GABOR_NOISE_H