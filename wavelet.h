#ifndef WAVELET_H
#define WAVELET_H

#include <cmath>
#include <vector>
#include "vec3.h"
#include "rtweekend.h"

class wavelet {
public:
    wavelet(int size = 64);

    double noise(const point3& p) const;
    double turb(const point3& p, int depth = 7) const;
    double projected_noise(const point3& p, const vec3& normal) const;

private:
    static constexpr int TILE_SIZE = 64;
    static constexpr int ARAD = 16;
    std::vector<float> noiseTileData;

    int mod(int x, int n) const;
    float random_gaussian() const;
    void generate_noise_tile();

    void downsample(const float* from, float* to, int stride);
    void upsample(const float* from, float* to, int stride);
    double evaluate_noise(const point3& p) const;
    double evaluate_projected_noise(const point3& p, const vec3& normal) const;
};

#endif
