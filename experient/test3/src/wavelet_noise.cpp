#include "wavelet_noise.h"
#include <algorithm>

WaveletNoise::WaveletNoise(int tileSize) : tileSize(tileSize) {
    NoiseUtils::log("WaveletNoise: Initializing with tile size " + std::to_string(tileSize));
    generateNoiseTile();
}

void WaveletNoise::generateNoiseTile() {
    NoiseUtils::log("WaveletNoise: Generating noise tile");
    
    // Step 1: Create random noise
    noiseTile.resize(tileSize, std::vector<float>(tileSize));
    for (int y = 0; y < tileSize; y++) {
        for (int x = 0; x < tileSize; x++) {
            noiseTile[y][x] = NoiseUtils::random_gaussian();
        }
    }
    
    // Step 2 & 3: Downsample and upsample
    auto downsampled = noiseTile;
    downsample2D(downsampled);
    
    auto upsampled = downsampled;
    upsample2D(upsampled);
    
    // Step 4: Subtract to get band-limited noise
    for (int y = 0; y < tileSize; y++) {
        for (int x = 0; x < tileSize; x++) {
            noiseTile[y][x] -= upsampled[y][x];
        }
    }
    
    NoiseUtils::log("WaveletNoise: Noise tile generation complete");
}

void WaveletNoise::downsample2D(std::vector<std::vector<float>>& data) {
    int size = data.size();
    std::vector<std::vector<float>> result(size/2, std::vector<float>(size/2));
    
    // Simple box filter for downsampling
    for (int y = 0; y < size/2; y++) {
        for (int x = 0; x < size/2; x++) {
            float sum = 0.0f;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    sum += data[y*2 + dy][x*2 + dx];
                }
            }
            result[y][x] = sum / 4.0f;
        }
    }
    
    // Resize original
    data = result;
}

void WaveletNoise::upsample2D(std::vector<std::vector<float>>& data) {
    int size = data.size();
    std::vector<std::vector<float>> result(size*2, std::vector<float>(size*2));
    
    // Bilinear interpolation for upsampling
    for (int y = 0; y < size*2; y++) {
        for (int x = 0; x < size*2; x++) {
            float fx = x / 2.0f;
            float fy = y / 2.0f;
            int ix = static_cast<int>(fx);
            int iy = static_cast<int>(fy);
            float dx = fx - ix;
            float dy = fy - iy;
            
            ix = std::min(ix, size-2);
            iy = std::min(iy, size-2);
            
            float v00 = data[iy][ix];
            float v10 = data[iy][ix+1];
            float v01 = data[iy+1][ix];
            float v11 = data[iy+1][ix+1];
            
            float v0 = NoiseUtils::lerp(v00, v10, dx);
            float v1 = NoiseUtils::lerp(v01, v11, dx);
            result[y][x] = NoiseUtils::lerp(v0, v1, dy);
        }
    }
    
    data = result;
}

float WaveletNoise::evaluate2D(float x, float y) {
    // Wrap coordinates
    x = fmod(x, 1.0f);
    y = fmod(y, 1.0f);
    if (x < 0) x += 1.0f;
    if (y < 0) y += 1.0f;
    
    return evaluate(x * tileSize, y * tileSize, noiseTile);
}

float WaveletNoise::evaluate(float x, float y, const std::vector<std::vector<float>>& tile) {
    int ix = static_cast<int>(x);
    int iy = static_cast<int>(y);
    float fx = x - ix;
    float fy = y - iy;
    
    ix %= tileSize;
    iy %= tileSize;
    int ix1 = (ix + 1) % tileSize;
    int iy1 = (iy + 1) % tileSize;
    
    // Bilinear interpolation
    float v00 = tile[iy][ix];
    float v10 = tile[iy][ix1];
    float v01 = tile[iy1][ix];
    float v11 = tile[iy1][ix1];
    
    float v0 = NoiseUtils::lerp(v00, v10, fx);
    float v1 = NoiseUtils::lerp(v01, v11, fx);
    
    return NoiseUtils::lerp(v0, v1, fy);
}