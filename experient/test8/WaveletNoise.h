#ifndef WAVELET_NOISE_H
#define WAVELET_NOISE_H

#include <vector>
#include <random>
#include <string>
#include <limits>
#include <iostream> // For stats output

// Statistical analysis structure
struct DataStats {
    float avg = 0.0f;
    float var = 0.0f;
    float min_val = std::numeric_limits<float>::max();
    float max_val = std::numeric_limits<float>::lowest();
    long long count_nan_inf = 0;
    float energy = 0.0f;  // Sum of squares
};

class WaveletNoise {
public:
    WaveletNoise(int tileSize, unsigned int seed = 0);
    ~WaveletNoise();

    // Generates the noise coefficients
    // Following Section 3.6 for separable filtering
    void generateNoiseTile2D();
    void generateNoiseTile3D();

    // Evaluation functions
    // Following Appendix 2, WNoise
    float evaluate2D(const float p[2]) const;
    float evaluate3D(const float p[3]) const;
    // Following Section 3.7 and Appendix 2, WProjectedNoise
    float evaluate3DProjected(const float p[3], const float normal[3]) const;

    // Debug and analysis
    DataStats calculateStats(const std::vector<float>& data, const std::string& name) const;
    const std::vector<float>& getNoiseCoefficients() const;
    int getTileSize() const;


private:
    int tileSizeN;
    std::vector<float> noiseCoefficients;
    unsigned int randomSeed;
    std::mt19937 rng;
    std::normal_distribution<float> gaussianDist;

    // Filter coefficients from Appendix 1
    static const int ARAD = 16;
    static const float A_COEFFS[2 * ARAD]; // Downsampling filter
    static const float P_COEFFS[4];        // Upsampling filter

    // Helper functions
    int Mod(int x, int n) const;
    void downsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride);
    void upsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride);
};

#endif // WAVELET_NOISE_H