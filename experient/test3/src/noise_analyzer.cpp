#include "noise_analyzer.h"
#include "perlin_noise.h"
#include "sparse_convolution_noise.h"
#include "wavelet_noise.h"
#include "anisotropic_noise.h"
#include "better_gradient_noise.h"
#include "gabor_noise.h"
#include <filesystem>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

NoiseAnalyzer::NoiseAnalyzer(int imageSize) : imageSize(imageSize) {
    NoiseUtils::log("NoiseAnalyzer: Initialized with image size " + 
                   std::to_string(imageSize));
}

void NoiseAnalyzer::analyzeNoise(std::shared_ptr<NoiseBase> noise, float scale) {
    NoiseUtils::log("NoiseAnalyzer: Analyzing " + noise->getName());
    
    AnalysisResult result;
    result.noiseName = noise->getName();
    
    // Generate noise image
    NoiseUtils::log("  Generating noise image...");
    result.noiseImage = noise->generateImage(imageSize, imageSize, scale);
    
    // Compute power spectrum
    NoiseUtils::log("  Computing power spectrum...");
    result.powerSpectrum = noise->getPowerSpectrum(imageSize, imageSize, scale);
    
    // Compute amplitude distribution
    NoiseUtils::log("  Computing amplitude distribution...");
    result.amplitudeDistribution = noise->getAmplitudeDistribution(imageSize, imageSize, scale);
    
    // Compute radial power spectrum
    NoiseUtils::log("  Computing radial power spectrum...");
    result.radialPowerSpectrum = noise->getRadialPowerSpectrum(imageSize, imageSize, scale);
    
    results.push_back(result);
    
    NoiseUtils::log("NoiseAnalyzer: Analysis complete for " + noise->getName());
}

void NoiseAnalyzer::analyzeAllNoises() {
    NoiseUtils::log("NoiseAnalyzer: Starting analysis of all noise types");
    
    // Create noise instances
    std::vector<std::shared_ptr<NoiseBase>> noises = {
        std::make_shared<PerlinNoise>(),
        std::make_shared<SparseConvolutionNoise>(),
        std::make_shared<WaveletNoise>(),
        std::make_shared<AnisotropicNoise>(),
        std::make_shared<BetterGradientNoise>(),
        std::make_shared<GaborNoise>()
    };
    
    // Analyze each noise
    for (auto& noise : noises) {
        analyzeNoise(noise);
    }
    
    NoiseUtils::log("NoiseAnalyzer: All analyses complete");
}

void NoiseAnalyzer::plotAmplitudeDistribution(const std::vector<float>& distribution, 
                                             const std::string& title) {
    int histWidth = 512;
    int histHeight = 400;
    int binWidth = histWidth / distribution.size();
    
    cv::Mat histImage(histHeight, histWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Find max value for normalization
    float maxVal = *std::max_element(distribution.begin(), distribution.end());
    
    // Draw histogram bars
    for (size_t i = 0; i < distribution.size(); i++) {
        int height = static_cast<int>(distribution[i] / maxVal * (histHeight - 20));
        cv::rectangle(histImage,
                     cv::Point(i * binWidth, histHeight - height - 10),
                     cv::Point((i + 1) * binWidth - 1, histHeight - 10),
                     cv::Scalar(0, 0, 255),
                     cv::FILLED);
    }
    
    // Add Gaussian reference
    std::vector<cv::Point> gaussianPoints;
    for (size_t i = 0; i < distribution.size(); i++) {
        float x = (i - distribution.size()/2.0f) / (distribution.size()/6.0f);
        float y = exp(-0.5f * x * x) / sqrt(2 * M_PI);
        int pixelY = histHeight - 10 - static_cast<int>(y * (histHeight - 20) / 0.4f);
        gaussianPoints.push_back(cv::Point(i * binWidth + binWidth/2, pixelY));
    }
    
    cv::polylines(histImage, gaussianPoints, false, cv::Scalar(0, 255, 0), 2);
    
    cv::imshow(title, histImage);
}

void NoiseAnalyzer::plotRadialPowerSpectrum(const std::vector<float>& spectrum, 
                                           const std::string& title) {
    int plotWidth = 512;
    int plotHeight = 400;
    
    cv::Mat plotImage(plotHeight, plotWidth, CV_8UC3, cv::Scalar(255, 255, 255));
    
    // Find max value for normalization
    float maxVal = *std::max_element(spectrum.begin(), spectrum.end());
    
    // Draw spectrum
    std::vector<cv::Point> points;
    for (size_t i = 0; i < spectrum.size() && i < plotWidth; i++) {
        int x = static_cast<int>(i * plotWidth / spectrum.size());
        int y = plotHeight - 10 - static_cast<int>(spectrum[i] / maxVal * (plotHeight - 20));
        points.push_back(cv::Point(x, y));
    }
    
    cv::polylines(plotImage, points, false, cv::Scalar(255, 0, 0), 2);
    
    cv::imshow(title, plotImage);
}

void NoiseAnalyzer::saveResults(const std::string& outputDir) {
    NoiseUtils::log("NoiseAnalyzer: Saving results to " + outputDir);
    
    // Create output directory
    std::filesystem::create_directories(outputDir);
    
    for (const auto& result : results) {
        std::string baseName = outputDir + "/" + result.noiseName;
        
        // Save noise image
        cv::Mat noiseImage8U;
        result.noiseImage.convertTo(noiseImage8U, CV_8U, 255.0);
        cv::imwrite(baseName + "_noise.png", noiseImage8U);
        NoiseUtils::log("  Saved: " + baseName + "_noise.png");
        
        // Save power spectrum
        cv::Mat spectrum8U;
        cv::normalize(result.powerSpectrum, spectrum8U, 0, 255, cv::NORM_MINMAX);
        spectrum8U.convertTo(spectrum8U, CV_8U);
        cv::imwrite(baseName + "_spectrum.png", spectrum8U);
        NoiseUtils::log("  Saved: " + baseName + "_spectrum.png");
        
        // Plot and save amplitude distribution
        plotAmplitudeDistribution(result.amplitudeDistribution, 
                                result.noiseName + " - Amplitude Distribution");
        
        // Plot and save radial power spectrum
        plotRadialPowerSpectrum(result.radialPowerSpectrum, 
                              result.noiseName + " - Radial Power Spectrum");
    }
    
    NoiseUtils::log("NoiseAnalyzer: All results saved");
}