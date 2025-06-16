#include <iostream>
#include <memory>
#include <opencv2/highgui.hpp>
#include "noise_analyzer.h"
#include "utils.h"

int main(int argc, char* argv[]) {
    NoiseUtils::log("=== Noise Analysis Tool ===");
    NoiseUtils::log("Recreating Figure 12 from the paper");
    
    try {
        // Create analyzer
        NoiseAnalyzer analyzer(512);
        
        // Analyze all noise types
        NoiseUtils::log("\nStarting noise analysis...");
        analyzer.analyzeAllNoises();
        
        // Save results
        std::string outputDir = "noise_analysis_results";
        if (argc > 1) {
            outputDir = argv[1];
        }
        
        NoiseUtils::log("\nSaving results...");
        analyzer.saveResults(outputDir);
        
        NoiseUtils::log("\nAnalysis complete! Results saved to: " + outputDir);
        NoiseUtils::log("Press any key to close visualization windows...");
        
        cv::waitKey(0);
        cv::destroyAllWindows();
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}