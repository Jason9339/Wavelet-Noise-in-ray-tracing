#ifndef WAVELET_NOISE_H
#define WAVELET_NOISE_H

#include <vector>
#include <random> // For random number generation

class WaveletNoise {
public:
    WaveletNoise(int tileSize, unsigned int seed = 0);
    ~WaveletNoise();

    // Generates the noise coefficients for a single band
    void generateNoiseTile();
    void generateNoiseTile2D();

    float evaluate3DProjected(const float p[3], const float normal[3]) const;

    // Evaluates the 2D noise band N(x,y) at normalized coordinates (u,v) in [0,1)x[0,1)
    // Assumes tile coefficients are for N(x,y) = sum n_ij B(2x-i)B(2y-j)
    // where x,y are coordinates in the tile's own space (e.g. 0 to tileSize-1)
    // So, u = x/tileSize, v = y/tileSize
    float evaluate2D(float u, float v) const;

    // Evaluates 3D noise (as per paper's WNoise function)
    // p[3] are coordinates in the object/world space that need to be scaled
    // for the current noise band.
    // This function evaluates ONE band of noise.
    // The final M(x) = sum w_b N(2^b * x) is done outside.
    float evaluate3D(const float p[3]) const;

    enum class DebugStep {
        R_INITIAL,
        AFTER_X_FILTER, // R_ds_us_x in temp2
        AFTER_Y_FILTER, // R_ds_us_xy in temp1
        AFTER_Z_FILTER, // R_ds_us_xyz (R_downarrow_uparrow) in temp2
        N_COEFFS_PRE_CORRECTION,
        N_COEFFS_FINAL
    };
    void saveIntermediateData(const std::vector<float>& data, int dim_n, DebugStep step, const std::string& base_filename = "wn_step_") const;
    float calculateTotalEnergy(const std::vector<float>& data) const;

    const std::vector<float>& getNoiseCoefficients() const;
    int getTileSize() const;

private:
    int tileSizeN; // Renamed to avoid conflict with member
    std::vector<float> noiseCoefficients; // Stores n_i for 1D, n_ij for 2D, n_ijk for 3D
    unsigned int randomSeed;
    std::mt19937 rng; // Random number generator
    std::normal_distribution<float> gaussianDist;

    // Filter coefficients (from paper Appendix 1)
    static const int ARAD = 16;
    static const float A_COEFFS[2 * ARAD]; // Analysis (downsampling)
    static const float P_COEFFS[4];        // Synthesis (upsampling)

    // Helper: Modulo function consistent with paper's Mod
    int Mod(int x, int n) const;

    // 1D downsampling and upsampling operations (applied separably)
    void downsample1D(const std::vector<float>& from, std::vector<float>& to, int n, int stride);
    void upsample1D(const std::vector<float>& from_coarse, std::vector<float>& to_fine, int n_fine, int stride);

    // For 3D version
    void generateNoiseTile3D(); // Paper's primary generation function
};

#endif // WAVELET_NOISE_H