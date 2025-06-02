#ifndef PERLIN_H
#define PERLIN_H

#include "rtweekend.h"
#include "vec3.h"
#include <cmath>

using point3 = vec3;

inline double perlin_fade(double t) {
    return t * t * t * (t * (t * 6 - 15) + 10);
}

inline double perlin_lerp(double t, double a, double b) {
    return a + t * (b - a);
}

const vec3 grad3[12] = {
    vec3(1,1,0), vec3(-1,1,0), vec3(1,-1,0), vec3(-1,-1,0),
    vec3(1,0,1), vec3(-1,0,1), vec3(1,0,-1), vec3(-1,0,-1),
    vec3(0,1,1), vec3(0,-1,1), vec3(0,1,-1), vec3(0,-1,-1)
};

inline double grad(int hash, double x, double y, double z) {
    int h = hash % 12;
    const vec3& g = grad3[h];
    return g.x() * x + g.y() * y + g.z() * z;
}

class perlin {
  public:
    static const int turbulence_depth = 20;

    perlin() {}

    double noise(const point3& p) const {
        int X = static_cast<int>(std::floor(p.x())) & 255;
        int Y = static_cast<int>(std::floor(p.y())) & 255;
        int Z = static_cast<int>(std::floor(p.z())) & 255;

        double x = p.x() - std::floor(p.x());
        double y = p.y() - std::floor(p.y());
        double z = p.z() - std::floor(p.z());

        double u = perlin_fade(x);
        double v = perlin_fade(y);
        double w = perlin_fade(z);

        int A  = (perm[X] + Y) & 255;
        int B  = (perm[X + 1] + Y) & 255;
        int AA = (perm[A] + Z) & 255;
        int BA = (perm[B] + Z) & 255;
        int AB = (perm[A + 1] + Z) & 255;
        int BB = (perm[B + 1] + Z) & 255;

        return perlin_lerp(w,
            perlin_lerp(v,
                perlin_lerp(u, grad(perm[AA], x, y, z), grad(perm[BA], x - 1, y, z)),
                perlin_lerp(u, grad(perm[AB], x, y - 1, z), grad(perm[BB], x - 1, y - 1, z))
            ),
            perlin_lerp(v,
                perlin_lerp(u, grad(perm[AA + 1], x, y, z - 1), grad(perm[BA + 1], x - 1, y, z - 1)),
                perlin_lerp(u, grad(perm[AB + 1], x, y - 1, z - 1), grad(perm[BB + 1], x - 1, y - 1, z - 1))
            )
        );
    }

    double turbulence_noise(const point3& p) const {
        double accum = 0.0;
        point3 temp_p = p;
        double weight = 1.0;

        for (int i = 0; i < turbulence_depth; i++) {
            accum += weight * noise(temp_p);
            temp_p *= 2.0;
            weight *= 0.5;
        }

        return accum;
    }

  private:
    static const int perm[512];
};

const int perlin::perm[512] = {
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,
    140,36,103,30,69,142,8,99,37,240,21,10,23,190,6,148,
    247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,
    57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,
    60,211,133,230,220,105,92,41,55,46,245,40,244,102,143,54,
    65,25,63,161,1,216,80,73,209,76,132,187,208,89,18,169,
    200,196,135,130,116,188,159,86,164,100,109,198,173,186,3,64,
    52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,
    207,182,16,101,179,19,59,12,184,251,138,114,106,253,49,4,
    150,242,189,107,222,239,170,162,78,221,153,163,45,14,213,195,
    224,50,192,243,2,183,227,119,191,145,235,205,98,112,232,115,
    246,155,97,254,61,24,104,223,151,160,137,91,90,15,131,13,
    201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,
    37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,
    94,252,219,203,117,35,11,32,57,177,33,88,237,149,56,87,
    174,20,125,136,171,168,68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,
    55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,
    209,76,132,187,208,89,18,169,200,196,135,130,116,188,159,86,
    164,100,109,198,173,186,3,64,52,217,226,250,124,123,5,202,
    38,147,118,126,255,82,85,212,207,182,16,101,179,19,59,12,
    184,251,138,114,106,253,49,4,150,242,189,107,222,239,170,162,
    78,221,153,163,45,14,213,195,224,50,192,243,2,183,227,119,
    191,145,235,205,98,112,232,115,246,155,97,254,61,24,104,223
};

#endif
