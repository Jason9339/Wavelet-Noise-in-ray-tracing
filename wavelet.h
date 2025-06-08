#ifndef WAVELET_NOISE_H
#define WAVELET_NOISE_H

#include "rtweekend.h" // For random_double, vec3, point3
#include "vec3.h"
#include <vector>
#include <cmath>
#include <numeric> // for std::iota
#include <algorithm> // for std::shuffle
#include <random>    // for std::mt19937, std::normal_distribution

// Forward declaration
class perlin; // If needed for comparison or hybrid approaches later, not directly for wavelet

using point3 = vec3;

class wavelet_noise {
public:
    // Constructor: n is the tile size (must be even, will be adjusted if odd)
    // olap is for tile meshing, not used in core generation here but kept for paper consistency.
    wavelet_noise(int tile_size_ = 32); // Smaller default for faster init, paper often implies larger

    // Evaluates non-projected 3D wavelet noise at point p for a single band (highest frequency)
    double noise_3d(const point3& p) const;

    // Evaluates non-projected 3D fractal wavelet noise
    double fractal_noise_3d(const point3& p, int octaves = 7, double persistence = 0.5) const;

    // Evaluates 3D fractal wavelet noise projected onto a 2D surface defined by normal
    double projected_fractal_noise_3d(const point3& p, const vec3& normal, int octaves = 7, double persistence = 0.5) const;

private:
    std::vector<double> noise_tile_data;
    int tile_size;

    // Filter coefficients (from Appendix 1)
    static const std::vector<double> aCoeffs; // Downsampling (analysis)
    static const std::vector<double> pCoeffs; // Upsampling (synthesis/refinement)
    static const int ARAD = 16; // Radius for aCoeffs

    // Helper: Modulo for periodic boundaries
    static int Mod(int x, int n);

    // Helper: Gaussian noise generator
    static double gaussian_noise(std::mt19937& gen);

    // Appendix 1: Tile Generation
    void GenerateNoiseTile(int n, int olap = 0); // olap not used in core generation here
    void Downsample(const std::vector<double>& from_coeffs, std::vector<double>& to_coeffs, int n_samples, int stride, const std::vector<double>& current_tile_data) const;
    void Upsample(const std::vector<double>& from_coeffs, std::vector<double>& to_coeffs, int n_samples, int stride, const std::vector<double>& current_tile_half_data) const;


    // Appendix 2: Noise Evaluation
    // Evaluates a single band of non-projected 3D noise
    double WNoise(const point3& p_scaled) const;
    // Evaluates a single band of 3D noise projected onto 2D
    double WProjectedNoise(const point3& p_scaled, const vec3& normal_scaled) const;
    // Multiband noise (called by public fractal_noise methods)
    double WMultibandNoise(const point3& p, const vec3* normal, int nbands, double persistence) const;

    // Helper for B-spline evaluation (from WNoise and WProjectedNoise)
    // Weights for quadratic B-spline: w[0], w[1], w[2] for t-1, t, t+1
    // p_val is the fractional coordinate (0 to 1)
    void evaluate_quadratic_bspline_weights(double p_val, double weights[3]) const;

};

#endif