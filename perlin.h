#ifndef PERLIN_H
#define PERLIN_H

#include "rtweekend.h"
#include "vec3.h"
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

using point3 = vec3;

class perlin {
private:
    std::vector<int> p;

    double fade(double t) const noexcept {
        return t * t * t * (t * (t * 6 - 15) + 10);
    }

    double lerp(double t, double a, double b) const noexcept {
        return a + t * (b - a);
    }

    double grad(int hash, double x, double y, double z) const noexcept {
        const int h = hash & 15;
        const double u = h < 8 ? x : y;
        const double v = h < 4 ? y : h == 12 || h == 14 ? x : z;
        return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
    }

public:
    explicit perlin(unsigned int seed = std::mt19937::default_seed) {
        p.resize(256);
        std::iota(p.begin(), p.end(), 0);
        std::shuffle(p.begin(), p.end(), std::mt19937(seed));
        p.insert(p.end(), p.begin(), p.end());
    }

    // 主要噪聲函數 - 與 experient 同步
    double noise(double x, double y, double z) const noexcept {
        const int X = static_cast<int>(std::floor(x)) & 255;
        const int Y = static_cast<int>(std::floor(y)) & 255;
        const int Z = static_cast<int>(std::floor(z)) & 255;

        x -= std::floor(x);
        y -= std::floor(y);
        z -= std::floor(z);

        const double u = fade(x);
        const double v = fade(y);
        const double w = fade(z);

        const int A = p[X] + Y, AA = p[A] + Z, AB = p[A + 1] + Z;
        const int B = p[X + 1] + Y, BA = p[B] + Z, BB = p[B + 1] + Z;

        return lerp(w, lerp(v, lerp(u, grad(p[AA], x, y, z), grad(p[BA], x - 1, y, z)),
                               lerp(u, grad(p[AB], x, y - 1, z), grad(p[BB], x - 1, y - 1, z))),
                       lerp(v, lerp(u, grad(p[AA + 1], x, y, z - 1), grad(p[BA + 1], x - 1, y, z - 1)),
                               lerp(u, grad(p[AB + 1], x, y - 1, z - 1), grad(p[BB + 1], x - 1, y - 1, z - 1))));
    }
    
    // 2D 噪聲（使用 z=0）
    double noise(double x, double y) const noexcept {
        return noise(x, y, 0.0);
    }

    // 3D 向量版本 - 為 ray tracing 兼容性
    double noise(const point3& p) const {
        return noise(p.x(), p.y(), p.z());
    }

    // 分形噪聲 - 多層疊加（預設 6 個 octaves）
    double fractal_noise(const point3& p) const {
        double result = 0.0;
        double amplitude = 1.0;
        double frequency = 1.0;
        double maxValue = 0.0;
        const int octaves = 6;

        for (int i = 0; i < octaves; i++) {
            result += noise(p.x() * frequency, p.y() * frequency, p.z() * frequency) * amplitude;
            maxValue += amplitude;
            amplitude *= 0.5;
            frequency *= 2.0;
        }

        return result / maxValue;
    }
};

#endif
