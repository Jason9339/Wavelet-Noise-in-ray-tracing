#include "WaveletNoise.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

// ====================================================================
// MODIFIED FOR INTEGER GRID SAMPLING
// ====================================================================
void generate2DOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    // Var(N(x))_2D ≈ Var(n_i)_2D * 0.265 ≈ 0.742875 * 0.265 ≈ 0.19686
    const float inv_stddev_2d = 1.0f / std::sqrt(0.19686f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            // *** INTEGER GRID SAMPLING ***
            // Directly use pixel coordinates. The input to evaluate2D will be integers.
            // We scale by 2 to see finer details, this is an arbitrary choice.
            float p[2] = {
                static_cast<float>(x) / 2.0f,
                static_cast<float>(y) / 2.0f
            };
            
            float value = noise.evaluate2D(p) * inv_stddev_2d;
            image[y * imageSize + x] = value;
        }
    }
    
    // Normalize for visualization
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range > 0) for (float& val : image) val = (val - minVal) / range;
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Generated pure 2D octave " << octave << " noise (Integer Grid): " << outputFile << std::endl;
}

// ====================================================================
// MODIFIED FOR INTEGER GRID SAMPLING
// ====================================================================
void generate3DSlicedOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);

    // Var(N(x))_3D ≈ Var(n_i)_3D * 0.210 ≈ 0.876291 * 0.210 ≈ 0.18402
    const float inv_stddev_3d = 1.0f / std::sqrt(0.18402f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            // *** INTEGER GRID SAMPLING ***
            float p[3] = {
                static_cast<float>(x) / 2.0f,
                static_cast<float>(y) / 2.0f,
                1.0f // Keep z constant
            };

            float value = noise.evaluate3D(p) * inv_stddev_3d;
            image[y * imageSize + x] = value;
        }
    }

    // Normalize for visualization
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range > 0) for (float& val : image) val = (val - minVal) / range;
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Generated 3D sliced octave " << octave << " noise (Integer Grid): " << outputFile << std::endl;
}

// ====================================================================
// MODIFIED FOR INTEGER GRID SAMPLING
// ====================================================================
void generate3DProjectedOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    float normal[3] = {0.0f, 0.0f, 1.0f};
    const float inv_stddev_3d_proj = 1.0f / std::sqrt(0.296f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            // *** INTEGER GRID SAMPLING ***
            float p[3] = {
                static_cast<float>(x) / 2.0f,
                static_cast<float>(y) / 2.0f,
                1.0f // The z-plane we are sampling on
            };
            
            float value = noise.evaluate3DProjected(p, normal) * inv_stddev_3d_proj;
            image[y * imageSize + x] = value;
        }
    }
    
    // Normalize for visualization
    float minVal = *std::min_element(image.begin(), image.end());
    float maxVal = *std::max_element(image.begin(), image.end());
    float range = maxVal - minVal;
    if (range > 0) for (float& val : image) val = (val - minVal) / range;
    
    std::ofstream outFile(outputFile, std::ios::binary);
    outFile.write(reinterpret_cast<const char*>(image.data()), image.size() * sizeof(float));
    outFile.close();
    
    std::cout << "Generated 3D projected octave " << octave << " noise (Integer Grid): " << outputFile << std::endl;
}


int main() {
    std::cout << "=== Wavelet Noise Verification (Integer Grid Sampling Test) ===" << std::endl;
    
    // For integer grid test, octave doesn't make sense anymore as we are not scaling.
    // We will just generate one set of images.
    const int IMAGE_SIZE = 256;
    const int TILE_SIZE = 128; // Tile size should be large enough
    const int a_single_octave = 0; // Octave is now irrelevant.
    
    std::cout << "\n--- Generating Pure 2D Noise Patterns ---" << std::endl;
    WaveletNoise noise2D(TILE_SIZE, 12345);
    noise2D.generateNoiseTile2D();
    generate2DOctaveBandNoise(IMAGE_SIZE, a_single_octave, "wavelet_noise_2D_octave_intgrid.raw", noise2D);

    std::cout << "\n--- Generating 3D Sliced & Projected Noise Patterns ---" << std::endl;
    WaveletNoise noise3D(TILE_SIZE, 12345);
    noise3D.generateNoiseTile3D();
    generate3DSlicedOctaveBandNoise(IMAGE_SIZE, a_single_octave, "wavelet_noise_3Dsliced_octave_intgrid.raw", noise3D);
    generate3DProjectedOctaveBandNoise(IMAGE_SIZE, a_single_octave, "wavelet_noise_3Dprojected_octave_intgrid.raw", noise3D);

    std::cout << "\n=== Generation Complete ===" << std::endl;
    std::cout << "Please run the special analysis script 'analyze_intgrid.py'." << std::endl;
    
    return 0;
}