#ifndef WAVELET_NOISE_H
#define WAVELET_NOISE_H

#include <vector>
#include <random>
#include <string>
#include <limits>

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
    void generateNoiseTile();
    void generateNoiseTile3D();

    // Evaluation functions
    float evaluate3D(const float p[3]) const;
    float evaluate3DProjected(const float p[3], const float normal[3]) const;

    // Debug and analysis
    enum class DebugStep {
        R_INITIAL,
        AFTER_X_FILTER,
        AFTER_Y_FILTER,
        AFTER_Z_FILTER,
        N_COEFFS_PRE_CORRECTION,
        N_COEFFS_FINAL
    };
    
    void saveIntermediateData(const std::vector<float>& data, int dim_n, DebugStep step, 
                             const std::string& base_filename = "wn_step_") const;
    float calculateTotalEnergy(const std::vector<float>& data) const;
    DataStats calculateStats(const std::vector<float>& data, const std::string& name) const;

    const std::vector<float>& getNoiseCoefficients() const;
    int getTileSize() const;

private:
    int tileSizeN;
    std::vector<float> noiseCoefficients;
    unsigned int randomSeed;
    std::mt19937 rng;
    std::normal_distribution<float> gaussianDist;

    // Filter coefficients
    static const int ARAD = 16;
    static const float A_COEFFS[2 * ARAD];
    static const float P_COEFFS[4];

    // Helper functions
    int Mod(int x, int n) const;
    void downsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride);
    void upsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride);
};

#endif // WAVELET_NOISE_H