#ifndef UTILS_H
#define UTILS_H

#include <vector>
#include <random>
#include <complex>
#include <iostream>
#include <fstream>

namespace NoiseUtils {
    // Random number generation
    inline float random_uniform(float min = 0.0f, float max = 1.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(min, max);
        return dis(gen);
    }
    
    inline float random_gaussian(float mean = 0.0f, float stddev = 1.0f) {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::normal_distribution<float> dis(mean, stddev);
        return dis(gen);
    }
    
    // Interpolation functions
    inline float lerp(float a, float b, float t) {
        return a + t * (b - a);
    }
    
    inline float smoothstep(float t) {
        return t * t * (3.0f - 2.0f * t);
    }
    
    inline float quintic(float t) {
        return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
    }
    
    // 2D/3D vector operations
    struct Vec2 {
        float x, y;
        Vec2(float x = 0, float y = 0) : x(x), y(y) {}
        float dot(const Vec2& other) const { return x * other.x + y * other.y; }
        float length() const { return std::sqrt(x * x + y * y); }
    };
    
    struct Vec3 {
        float x, y, z;
        Vec3(float x = 0, float y = 0, float z = 0) : x(x), y(y), z(z) {}
        float dot(const Vec3& other) const { 
            return x * other.x + y * other.y + z * other.z; 
        }
        float length() const { return std::sqrt(x * x + y * y + z * z); }
    };
    
    // Logging
    inline void log(const std::string& message) {
        std::cout << "[LOG] " << message << std::endl;
    }
}

#endif // UTILS_H