#ifndef WAVELET_NOISE_H
#define WAVELET_NOISE_H

#include "noise_base.h"
#include <vector>

class WaveletNoise : public NoiseBase {
public:
    WaveletNoise(int tileSize = 128);
    
    float evaluate2D(float x, float y) override;
    std::string getName() const override { return "Wavelet Noise"; }
    
private:
    int tileSize;
    std::vector<std::vector<float>> noiseTile;
    
    void generateNoiseTile();
    void downsample2D(std::vector<std::vector<float>>& data);
    void upsample2D(std::vector<std::vector<float>>& data);
    float evaluate(float x, float y, const std::vector<std::vector<float>>& tile);
};

#endif // WAVELET_NOISE_H