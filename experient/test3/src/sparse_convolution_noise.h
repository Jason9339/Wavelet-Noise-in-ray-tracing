#ifndef SPARSE_CONVOLUTION_NOISE_H
#define SPARSE_CONVOLUTION_NOISE_H

#include "noise_base.h"
#include <vector>

class SparseConvolutionNoise : public NoiseBase {
public:
    SparseConvolutionNoise(float density = 10.0f, float kernelRadius = 0.1f);
    
    float evaluate2D(float x, float y) override;
    std::string getName() const override { return "Sparse Convolution Noise"; }
    
private:
    float density;
    float kernelRadius;
    
    struct ImpulsePoint {
        float x, y;
        float amplitude;
    };
    
    std::vector<ImpulsePoint> generateImpulses(int cellX, int cellY);
    float kernel(float distance);
    unsigned int hash(int x, int y);
};

#endif // SPARSE_CONVOLUTION_NOISE_H