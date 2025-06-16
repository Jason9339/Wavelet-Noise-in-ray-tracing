#include "noise_base.h"
#include <opencv2/opencv.hpp>

cv::Mat NoiseBase::generateImage(int width, int height, float scale) {
    NoiseUtils::log(getName() + ": Generating " + std::to_string(width) + "x" + 
                   std::to_string(height) + " image with scale " + std::to_string(scale));
    
    cv::Mat image(height, width, CV_32F);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float nx = x * scale / width;
            float ny = y * scale / height;
            image.at<float>(y, x) = evaluate2D(nx, ny);
        }
    }
    
    NoiseUtils::log(getName() + ": Image generation complete");
    return normalizeImage(image);
}

std::vector<float> NoiseBase::getAmplitudeDistribution(int width, int height, 
                                                      float scale, int bins) {
    NoiseUtils::log(getName() + ": Computing amplitude distribution");
    
    cv::Mat image = generateImage(width, height, scale);
    std::vector<float> histogram(bins, 0.0f);
    
    double minVal, maxVal;
    cv::minMaxLoc(image, &minVal, &maxVal);
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float val = image.at<float>(y, x);
            int bin = static_cast<int>((val - minVal) / (maxVal - minVal) * (bins - 1));
            histogram[bin]++;
        }
    }
    
    // Normalize histogram
    float total = width * height;
    for (auto& h : histogram) {
        h /= total;
    }
    
    NoiseUtils::log(getName() + ": Amplitude distribution computed");
    return histogram;
}

cv::Mat NoiseBase::getPowerSpectrum(int width, int height, float scale) {
    NoiseUtils::log(getName() + ": Computing power spectrum");
    
    cv::Mat image = generateImage(width, height, scale);
    cv::Mat padded;
    
    // Pad image to optimal size for DFT
    int m = cv::getOptimalDFTSize(image.rows);
    int n = cv::getOptimalDFTSize(image.cols);
    cv::copyMakeBorder(image, padded, 0, m - image.rows, 0, n - image.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0));
    
    // Perform DFT
    cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)};
    cv::Mat complexI;
    cv::merge(planes, 2, complexI);
    cv::dft(complexI, complexI);
    
    // Compute magnitude
    cv::split(complexI, planes);
    cv::magnitude(planes[0], planes[1], planes[0]);
    cv::Mat magI = planes[0];
    
    // Switch to logarithmic scale
    magI += cv::Scalar::all(1);
    cv::log(magI, magI);
    
    // Crop and rearrange quadrants
    magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));
    int cx = magI.cols/2;
    int cy = magI.rows/2;
    
    cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    NoiseUtils::log(getName() + ": Power spectrum computed");
    return magI;
}

std::vector<float> NoiseBase::getRadialPowerSpectrum(int width, int height, float scale) {
    NoiseUtils::log(getName() + ": Computing radial power spectrum");
    
    cv::Mat spectrum = getPowerSpectrum(width, height, scale);
    int centerX = spectrum.cols / 2;
    int centerY = spectrum.rows / 2;
    int maxRadius = std::min(centerX, centerY);
    
    std::vector<float> radialSpectrum(maxRadius, 0.0f);
    std::vector<int> counts(maxRadius, 0);
    
    for (int y = 0; y < spectrum.rows; y++) {
        for (int x = 0; x < spectrum.cols; x++) {
            int dx = x - centerX;
            int dy = y - centerY;
            int radius = static_cast<int>(std::sqrt(dx*dx + dy*dy));
            
            if (radius < maxRadius) {
                radialSpectrum[radius] += spectrum.at<float>(y, x);
                counts[radius]++;
            }
        }
    }
    
    // Average the values
    for (int i = 0; i < maxRadius; i++) {
        if (counts[i] > 0) {
            radialSpectrum[i] /= counts[i];
        }
    }
    
    NoiseUtils::log(getName() + ": Radial power spectrum computed");
    return radialSpectrum;
}

cv::Mat NoiseBase::normalizeImage(const cv::Mat& img) {
    cv::Mat normalized;
    cv::normalize(img, normalized, 0, 1, cv::NORM_MINMAX);
    return normalized;
}