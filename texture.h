#ifndef TEXTURE_H
#define TEXTURE_H

#include "vec3.h"
#include "perlin.h"
#include <cmath>
#include "rtweekend.h"
#include "WaveletNoise.h"
#include "color.h"

using color = vec3;
using point3 = vec3;

class texture {
  public:
    virtual ~texture() = default;
    virtual color value(double u, double v, const point3& p) const = 0;
};

class solid_color : public texture {
  public:
    solid_color(const color& albedo) : albedo(albedo) {}
    solid_color(double red, double green, double blue) : solid_color(color(red,green,blue)) {}

    color value(double u, double v, const point3& p) const override {
        return albedo;
    }

  private:
    color albedo;
};

class noise_texture : public texture {
  public:
    noise_texture(double scale, int octave = 4) : scale(scale), octave_level(octave) {}

    color value(double u, double v, const point3& p) const override {
        const float octave_scale = std::pow(2.0f, octave_level);
        point3 scaled_p = p * scale * octave_scale;
        double noise_val = noise.noise(scaled_p);
        noise_val = 0.5 * (1.0 + noise_val);
        return color(noise_val, noise_val, noise_val);
    }

  private:
    perlin noise;
    double scale;
    int octave_level;
};

class wavelet_texture : public texture {
  public:
    wavelet_texture(double scale = 1.0, int octave = 4, bool use_3d = true) 
        : scale(scale), octave_level(octave), use_3d_noise(use_3d) {
        const int TILE_SIZE = 128;
        const unsigned int SEED = 12345;
        
        noise_2d = std::make_unique<WaveletNoise>(TILE_SIZE, SEED);
        noise_2d->generateNoiseTile2D();
        
        if (use_3d_noise) {
            noise_3d = std::make_unique<WaveletNoise>(TILE_SIZE, SEED);
            noise_3d->generateNoiseTile3D();
        }
    }

    color value(double u, double v, const point3& p) const override {
        double noise_val;
        
        if (use_3d_noise && noise_3d) {
            float pos[3] = { 
                static_cast<float>(p.x() * scale), 
                static_cast<float>(p.y() * scale), 
                static_cast<float>(p.z() * scale) 
            };
            
            const float octave_scale = std::pow(2.0f, octave_level);
            pos[0] *= octave_scale * 2.0f;
            pos[1] *= octave_scale * 2.0f;
            pos[2] *= octave_scale * 2.0f;
            
            noise_val = noise_3d->evaluate3D(pos);
            
            const float inv_stddev_3d = 1.0f / std::sqrt(0.18402f);
            noise_val *= inv_stddev_3d;
        } else if (noise_2d) {
            float pos[2] = { 
                static_cast<float>(p.x() * scale), 
                static_cast<float>(p.y() * scale) 
            };
            
            const float octave_scale = std::pow(2.0f, octave_level);
            pos[0] *= octave_scale * 2.0f;
            pos[1] *= octave_scale * 2.0f;
            
            noise_val = noise_2d->evaluate2D(pos);
            
            const float inv_stddev_2d = 1.0f / std::sqrt(0.19686f);
            noise_val *= inv_stddev_2d;
        } else {
            noise_val = 0.0;
        }
        
        noise_val = 0.5 * (1.0 + std::clamp(noise_val / 4.0, -1.0, 1.0));
        
        return color(noise_val, noise_val, noise_val);
    }

  private:
    std::unique_ptr<WaveletNoise> noise_2d;
    std::unique_ptr<WaveletNoise> noise_3d;
    double scale;
    int octave_level;
    bool use_3d_noise;
};

#endif