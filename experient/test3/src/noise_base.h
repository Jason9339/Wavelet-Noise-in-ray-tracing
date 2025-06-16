#ifndef NOISE_BASE_H
#define NOISE_BASE_H

#include <vector>
#include <opencv2/opencv.hpp>
#include "utils.h"

class NoiseBase {
public:
    virtual ~NoiseBase() = default;
    
    // Generate 2D noise
    virtual float evaluate2D(float x, float y) = 0;
    
    // Generate noise image
    virtual cv::Mat generateImage(int width, int height, float scale = 1.0f);
    
    // Get amplitude distribution
    virtual std::vector<float> getAmplitudeDistribution(int width, int height, 
                                                       float scale = 1.0f,
                                                       int bins = 256);
    
    // Get power spectrum
    virtual cv::Mat getPowerSpectrum(int width, int height, float scale = 1.0f);
    
    // Get radially averaged power spectrum
    virtual std::vector<float> getRadialPowerSpectrum(int width, int height, 
                                                     float scale = 1.0f);
    
    // Get noise name for logging
    virtual std::string getName() const = 0;
    
protected:
    // Helper function to normalize image
    cv::Mat normalizeImage(const cv::Mat& img);
};

#endif // NOISE_BASE_H