#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>
#include <string>
#include <stdexcept> // For std::out_of_range

class Image {
public:
    int width, height;
    std::vector<float> pixels;

    Image(int w = 0, int h = 0);
    void resize(int w, int h);
    
    float get(int x, int y) const;
    void set(int x, int y, float value);

    float& at(int x, int y); // For direct modification
    const float& at(int x, int y) const; // For const access

    void normalize(float min_val = 0.0f, float max_val = 1.0f);
    void savePPM(const std::string& filename, bool log_scale = false, bool apply_normalize = true) const;
};

#endif // IMAGE_HPP