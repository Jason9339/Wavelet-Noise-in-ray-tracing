#include "Image.hpp"
#include <fstream>
#include <vector>
#include <algorithm> // For std::min_element, std::max_element
#include <cmath>     // For std::log, std::fabs
#include <iostream>  // For debug

Image::Image(int w, int h) : width(w), height(h) {
    if (w > 0 && h > 0) {
        pixels.resize(w * h, 0.0f);
    }
}

void Image::resize(int w, int h) {
    width = w;
    height = h;
    if (w > 0 && h > 0) {
        pixels.resize(w * h, 0.0f);
    } else {
        pixels.clear();
    }
}


float Image::get(int x, int y) const {
    // Clamped access
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    return pixels[y * width + x];
}

void Image::set(int x, int y, float value) {
    // Clamped access
    x = std::max(0, std::min(x, width - 1));
    y = std::max(0, std::min(y, height - 1));
    pixels[y * width + x] = value;
}


float& Image::at(int x, int y) {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        throw std::out_of_range("Image::at() coordinates out of bounds");
    }
    return pixels[y * width + x];
}

const float& Image::at(int x, int y) const {
    if (x < 0 || x >= width || y < 0 || y >= height) {
        throw std::out_of_range("Image::at() coordinates out of bounds");
    }
    return pixels[y * width + x];
}


void Image::normalize(float min_val_target, float max_val_target) {
    if (pixels.empty()) return;

    float current_min = pixels[0];
    float current_max = pixels[0];
    for (float p : pixels) {
        if (p < current_min) current_min = p;
        if (p > current_max) current_max = p;
    }

    // std::cout << "Normalization: Current min=" << current_min << ", max=" << current_max << std::endl;

    if (std::fabs(current_max - current_min) < 1e-6) { // Avoid division by zero if all pixels are same
        for (float& p : pixels) {
            p = min_val_target; // Or some other default, like (min_val + max_val)/2
        }
        return;
    }

    for (float& p : pixels) {
        p = min_val_target + (p - current_min) * (max_val_target - min_val_target) / (current_max - current_min);
    }
}


void Image::savePPM(const std::string& filename, bool log_scale, bool apply_normalize) const {
    if (width == 0 || height == 0) return;

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    ofs << "P5\n" << width << " " << height << "\n255\n";

    std::vector<float> temp_pixels = pixels; // Make a mutable copy

    if (log_scale) {
        for (float& p : temp_pixels) {
            p = std::log(1.0f + std::fabs(p)); // Use fabs for complex magnitudes
        }
    }
    
    if (apply_normalize) { // This part needs to be in Image itself to modify temp_pixels
        float current_min = temp_pixels[0];
        float current_max = temp_pixels[0];
        for (float p : temp_pixels) {
            if (p < current_min) current_min = p;
            if (p > current_max) current_max = p;
        }
        // std::cout << "PPM Save Normalization (" << filename << "): Min=" << current_min << ", Max=" << current_max << std::endl;

        if (std::fabs(current_max - current_min) < 1e-6) {
             for (float& p : temp_pixels) p = 0.0f; // All pixels will be 0
        } else {
            for (float& p : temp_pixels) {
                 p = (p - current_min) / (current_max - current_min);
            }
        }
    }


    for (float p_norm : temp_pixels) {
        unsigned char val = static_cast<unsigned char>(std::min(std::max(p_norm * 255.0f, 0.0f), 255.0f));
        ofs.put(val);
    }
    // std::cout << "Saved image: " << filename << std::endl;
}