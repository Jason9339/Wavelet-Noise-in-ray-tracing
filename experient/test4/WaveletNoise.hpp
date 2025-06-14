#ifndef WAVELET_NOISE_HPP
#define WAVELET_NOISE_HPP

#include "Image.hpp"
#include <vector>
#include <iostream> // For verification

class WaveletNoise {
public:
    WaveletNoise(int tileSize, unsigned int seed = 54321);
    float noise(float x, float y) const; // Coordinates are in tile units

private:
    int N_tile; // Tile size
    Image noise_coefficients; // Stores n_i from CD05 paper

    // CD05 quadratic B-spline filter coefficients
    static const std::vector<float> aCoeffs; // Downsampling analysis filter
    static const std::vector<float> pCoeffs; // Upsampling synthesis filter (refinement)

    void convolve1D(std::vector<float>& data, const std::vector<float>& filter_coeffs, bool downsample, bool circular);
    void generateTile(unsigned int seed);
};

#endif // WAVELET_NOISE_HPP