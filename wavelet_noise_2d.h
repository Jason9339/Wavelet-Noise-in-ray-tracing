#ifndef WAVELET_NOISE_2D_H
#define WAVELET_NOISE_2D_H

#include "rtweekend.h" // For random_double, vec3, point3 (though point3 less relevant here)
#include "vec2.h"      // We'll use vec2 for 2D points
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <random>

class wavelet_noise_2d {
public:
    // Constructor: n is the tile size (must be even)
    wavelet_noise_2d(int tile_size_ = 32);

    // Evaluates 2D wavelet noise at point p for a single band
    double noise_2d(const vec2& p) const;

    // Evaluates 2D fractal wavelet noise
    double fractal_noise_2d(const vec2& p, int octaves = 7, double persistence = 0.5) const;

private:
    std::vector<double> noise_tile_data_2d;
    int tile_size;

    // Filter coefficients (shared with 3D version, can be made common)
    static const std::vector<double> aCoeffs; // Downsampling (analysis)
    static const std::vector<double> pCoeffs; // Upsampling (synthesis/refinement)
    static const int ARAD = 16; // Radius for aCoeffs

    // Helper: Modulo for periodic boundaries
    static int Mod(int x, int n);

    // Helper: Gaussian noise generator
    static double gaussian_noise(std::mt19937& gen);

    // 2D Tile Generation
    void GenerateNoiseTile2D(int n, int olap = 0);
    // 1D Downsample/Upsample (can be shared with 3D if refactored, but kept separate for clarity now)
    void Downsample1D(const std::vector<double>& from_coeffs, std::vector<double>& to_coeffs, int n_elements) const;
    void Upsample1D(const std::vector<double>& from_coeffs_half_size, std::vector<double>& to_coeffs_full_size, int n_elements_full_size) const;

    // 2D Noise Evaluation for a single band
    double WNoise2D(const vec2& p_scaled) const;
    // Multiband noise (called by public fractal_noise method)
    double WMultibandNoise2D(const vec2& p, int nbands, double persistence) const;

    // Helper for B-spline evaluation (same as 3D, could be common utility)
    void evaluate_quadratic_bspline_weights(double p_val, double weights[3]) const;
};

#endif