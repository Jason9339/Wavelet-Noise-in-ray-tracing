#include "WaveletNoise.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <algorithm>

// Generates a 2D image of pure 2D band-limited noise at a specific octave.
void generate2DOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    float base_range = 4.0f;
    float octave_scale = std::pow(2.0f, octave);

    // Var(N(x))_2D ≈ Var(n_i)_2D * 0.265 ≈ 0.742875 * 0.265 ≈ 0.19686
    // The standard deviation is the square root of the variance.
    const float inv_stddev_2d = 1.0f / std::sqrt(0.19686f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            float p[2] = {2.0f * u * octave_scale, 2.0f * v * octave_scale};
            
            // Normalize the evaluated noise to have a variance of 1.
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
    
    std::cout << "Generated pure 2D octave " << octave << " noise: " << outputFile << std::endl;
}

// Generates a 2D image by slicing a 3D band-limited noise volume.
void generate3DSlicedOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    float base_range = 4.0f;
    float octave_scale = std::pow(2.0f, octave);

    // Var(N(x))_3D ≈ Var(n_i)_3D * 0.210 ≈ 0.876291 * 0.210 ≈ 0.18402
    // The standard deviation is the square root of the variance.
    const float inv_stddev_3d = 1.0f / std::sqrt(0.18402f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            float p[3] = {2.0f * u * octave_scale, 2.0f * v * octave_scale, 1.0f}; 
            
            // Normalize the evaluated noise to have a variance of 1.
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
    
    std::cout << "Generated 3D sliced octave " << octave << " noise: " << outputFile << std::endl;
}

void generate3DProjectedOctaveBandNoise(int imageSize, int octave, const std::string& outputFile, WaveletNoise& noise) {
    std::vector<float> image(imageSize * imageSize);
    
    float base_range = 4.0f;
    float octave_scale = std::pow(2.0f, octave);

    // *** CRITICAL FIX ***
    // Use an axis-aligned normal (0,0,1) to match the XY sampling plane.
    // This correctly tests the principle of Equation (32).
    float normal[3] = {0.0f, 0.0f, 1.0f};

    // Section 4.2: The variance of 3D projected noise is ~0.296.
    const float inv_stddev_3d_proj = 1.0f / std::sqrt(0.296f);

    for (int y = 0; y < imageSize; ++y) {
        for (int x = 0; x < imageSize; ++x) {
            float u = (static_cast<float>(x) / imageSize) * base_range;
            float v = (static_cast<float>(y) / imageSize) * base_range;
            
            // The sampling points lie on the XY plane (z is constant).
            float p[3] = {2.0f * u * octave_scale, 2.0f * v * octave_scale, 1.0f}; 
            
            // Use the projection evaluation function with the CORRECT normal.
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
    
    std::cout << "Generated 3D projected octave " << octave << " noise: " << outputFile << std::endl;
}


int main() {
    std::cout << "=== Wavelet Noise Implementation Verification (Final Corrected Version) ===" << std::endl;
    
    const int IMAGE_SIZE = 256;
    const int TILE_SIZE = 64; 
    
    std::cout << "\n--- Generating Pure 2D Noise Patterns ---" << std::endl;
    WaveletNoise noise2D(TILE_SIZE, 12345);
    noise2D.generateNoiseTile2D();
    noise2D.calculateStats(noise2D.getNoiseCoefficients(), "2D Noise Tile Coefficients");
    for (int octave = 3; octave <= 5; ++octave) {
        generate2DOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_2D_octave_" + std::to_string(octave) + ".raw", noise2D);
    }

    // --- 3D Sliced & Projected Noise Generation ---
    std::cout << "\n--- Generating 3D Sliced & Projected Noise Patterns ---" << std::endl;
    WaveletNoise noise3D(TILE_SIZE, 12345);
    noise3D.generateNoiseTile3D();
    noise3D.calculateStats(noise3D.getNoiseCoefficients(), "3D Noise Tile Coefficients");
    for (int octave = 3; octave <= 5; ++octave) {
        generate3DSlicedOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_3Dsliced_octave_" + std::to_string(octave) + ".raw", noise3D);
        generate3DProjectedOctaveBandNoise(IMAGE_SIZE, octave, "wavelet_noise_3Dprojected_octave_" + std::to_string(octave) + ".raw", noise3D);
    }

    std::cout << "\n=== Generation Complete ===" << std::endl;
    std::cout << "Projected noise generation has been corrected to use axis-alignment." << std::endl;
    std::cout << "Please run the 'analyze.py' script again." << std::endl;
    
    return 0;
}