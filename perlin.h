#ifndef PERLIN_H
#define PERLIN_H
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
#include <cmath>
#include <random>

using point3 = vec3;

inline double random_double() {
    static std::uniform_real_distribution<double> distribution(0.0, 1.0);
    static std::mt19937 generator;
    return distribution(generator);
}

inline double random_double(double min, double max) {
    return min + (max-min)*random_double();
}

inline int random_int(int min, int max) {
    return static_cast<int>(random_double(min, max+1));
}

inline vec3 random_vec3() {
    return vec3(random_double(), random_double(), random_double());
}

inline vec3 random_vec3(double min, double max) {
    return vec3(random_double(min,max), random_double(min,max), random_double(min,max));
}

inline double perlin_fade(double t) {
    // Quintic smoothstep: 6t^5 - 15t^4 + 10t^3
    return t * t * t * (t * (t * 6 - 15) + 10);
}

class perlin {
  public:
    perlin() {
        // Random gradient vectors
        for (int i = 0; i < point_count; i++) {
            randvec[i] = unit_vector(vec3(random_double(-1,1), random_double(-1,1), random_double(-1,1)));
        }

        // Use fixed permutation table for stability.  Copy the permutation array twice to fill perm_x, perm_y, and perm_z.
        for (int i = 0; i < point_count; ++i) {
            perm_x[i] = permutation[i];
            perm_y[i] = permutation[i];
            perm_z[i] = permutation[i];
        }
    }

    double noise(const point3& p) const {
        auto u = p.x() - std::floor(p.x());
        auto v = p.y() - std::floor(p.y());
        auto w = p.z() - std::floor(p.z());

        auto i = static_cast<int>(std::floor(p.x()));
        auto j = static_cast<int>(std::floor(p.y()));
        auto k = static_cast<int>(std::floor(p.z()));
        vec3 c[2][2][2];

        for (int di=0; di < 2; di++)
            for (int dj=0; dj < 2; dj++)
                for (int dk=0; dk < 2; dk++)
                    c[di][dj][dk] = randvec[
                        perm_x[(i+di) & 255] ^
                        perm_y[(j+dj) & 255] ^
                        perm_z[(k+dk) & 255]
                    ];

        return perlin_interp(c, u, v, w);
    }

    double turb(const point3& p, int depth) const {
        auto accum = 0.0;
        auto temp_p = p;
        auto weight = 1.0;

        for (int i = 0; i < depth; i++) {
            accum += weight * noise(temp_p);
            weight *= 0.5;
            temp_p *= 2.0;
        }

        return std::fabs(accum);
    }

  private:
    static const int point_count = 256;
    vec3 randvec[point_count];
    int perm_x[point_count];
    int perm_y[point_count];
    int perm_z[point_count];

    static double perlin_interp(const vec3 c[2][2][2], double u, double v, double w) {
        auto uu = perlin_fade(u);
        auto vv = perlin_fade(v);
        auto ww = perlin_fade(w);
        double accum = 0.0;

        for (int i=0; i<2; i++)
            for (int j=0; j<2; j++)
                for (int k=0; k<2; k++) {
                    vec3 weight_v(u - i, v - j, w - k);
                    accum += (i ? uu : 1 - uu) *
                             (j ? vv : 1 - vv) *
                             (k ? ww : 1 - ww) *
                             dot(c[i][j][k], weight_v);
                }

        return accum;
    }

    // Fixed permutation table from Ken Perlin (public domain)
    static const int permutation[256];
};

// Ken Perlin's original permutation (256 elements)
const int perlin::permutation[256] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,182,16,101,179,19,59,12,184,251,138,114,106,253,49,4,150,242,189,107,222,239,170,162,78,221,153,163,45,14,213,195,224,50,192,243,2,183,227,119,191,145,235,205,98,112,232,115,246,155,97,254,61,24,104,223
};
#endif