#ifndef TEXTURE_H
#define TEXTURE_H
//==============================================================================================
// Originally written in 2016 by Peter Shirley <ptrshrl@gmail.com>
//
// To the extent possible under law, the author(s) have dedicated all copyright and related and
// neighboring rights to this software to the public domain worldwide. This software is
// distributed without any warranty.
//
// You should have received a copy (see file COPYING.txt) of the CC0 Public Domain Dedication
// along with this software. If not, see <http://creativecommons.org/publicdomain/zero/1.0/>.
//==============================================================================================

#include "vec3.h"
#include "perlin.h"
#include <cmath>

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
    noise_texture(double scale) : scale(scale) {}

    color value(double u, double v, const point3& p) const override {
        // 對球體總是使用3D世界坐標，避免UV接縫問題
        double n = noise.turbulence_noise((p + vec3(100.123, 0.456, 87.789)) * scale);
        n = 0.5 + 0.5 * n;
        n = std::clamp(n, 0.0, 1.0);
        return color(n, n, n);
    }

  private:
    perlin noise;
    double scale;
};


#endif