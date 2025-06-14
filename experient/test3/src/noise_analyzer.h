#ifndef NOISE_ANALYZER_H
#define NOISE_ANALYZER_H

#include "noise_base.h"
#include <opencv2/opencv.hpp>
#include <memory>
#include <vector>

class NoiseAnalyzer {
public:
    NoiseAnalyzer(int imageSize = 512);
    
    // Analyze a specific noise function
    void analyzeNoise(std::shared_ptr<NoiseBase> noise, float scale = 32.0f);
    
    // Analyze all noise types
    void analyzeAllNoises();
    
    // Save results
    void saveResults(const std::string& outputDir);
    
private:
    int imageSize;
    
    struct AnalysisResult {
        std::string noiseName;
        cv::Mat noiseImage;
        cv::Mat powerSpectrum;
        std::vector<float> amplitudeDistribution;
        std::vector<float> radialPowerSpectrum;
    };
    
    std::vector<AnalysisResult> results;
    
    void plotAmplitudeDistribution(const std::vector<float>& distribution, 
                                  const std::string& title);
    void plotRadialPowerSpectrum(const std::vector<float>& spectrum, 
                               const std::string& title);
};

#endif // NOISE_ANALYZER_H