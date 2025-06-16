#ifndef VEC2_HPP
#define VEC2_HPP

#include <cmath>

struct Vec2 {
    float x, y;

    Vec2(float x_ = 0.0f, float y_ = 0.0f) : x(x_), y(y_) {}

    Vec2 operator+(const Vec2& other) const { return Vec2(x + other.x, y + other.y); }
    Vec2 operator-(const Vec2& other) const { return Vec2(x - other.x, y - other.y); }
    Vec2 operator*(float scalar) const { return Vec2(x * scalar, y * scalar); }
    float dot(const Vec2& other) const { return x * other.x + y * other.y; }
    float length() const { return std::sqrt(x * x + y * y); }
    Vec2 normalized() const {
        float l = length();
        if (l > 0) return Vec2(x / l, y / l);
        return Vec2(0,0);
    }
};

#endif // VEC2_HPP