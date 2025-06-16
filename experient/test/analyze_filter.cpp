#include <iostream>
#include <cmath>
#include <numeric>

void analyzeFilterCoefficients() {
    // Analysis filter (downsampling)
    float aCoeffs[32] = {
        0.000334f,-0.001528f, 0.000410f, 0.003545f,-0.000938f,-0.008233f, 0.002172f, 0.019120f,
        -0.005040f,-0.044412f, 0.011655f, 0.103311f,-0.025936f,-0.243780f, 0.033979f, 0.655340f,
        0.655340f, 0.033979f,-0.243780f,-0.025936f, 0.103311f, 0.011655f,-0.044412f,-0.005040f,
        0.019120f, 0.002172f,-0.008233f,-0.000938f, 0.003546f, 0.000410f,-0.001528f, 0.000334f
    };
    
    // Synthesis filter (upsampling)
    float pCoeffs[4] = { 0.25f, 0.75f, 0.75f, 0.25f };
    
    std::cout << "=== FILTER ANALYSIS ===" << std::endl;
    
    // Analyze analysis filter
    float a_sum = 0.0f;
    float a_sum_sq = 0.0f;
    float a_sum_even = 0.0f;
    float a_sum_odd = 0.0f;
    
    for (int i = 0; i < 32; i++) {
        a_sum += aCoeffs[i];
        a_sum_sq += aCoeffs[i] * aCoeffs[i];
        if (i % 2 == 0) {
            a_sum_even += aCoeffs[i];
        } else {
            a_sum_odd += aCoeffs[i];
        }
    }
    
    std::cout << "\nAnalysis filter (downsampling):" << std::endl;
    std::cout << "  Sum: " << a_sum << " (should be ~√2 = 1.414)" << std::endl;
    std::cout << "  Sum of squares: " << a_sum_sq << " (should be ~1.0)" << std::endl;
    std::cout << "  Sum even indices: " << a_sum_even << std::endl;
    std::cout << "  Sum odd indices: " << a_sum_odd << std::endl;
    std::cout << "  Even + odd: " << (a_sum_even + a_sum_odd) << std::endl;
    
    // Analyze synthesis filter
    float p_sum = 0.0f;
    float p_sum_sq = 0.0f;
    
    for (int i = 0; i < 4; i++) {
        p_sum += pCoeffs[i];
        p_sum_sq += pCoeffs[i] * pCoeffs[i];
    }
    
    std::cout << "\nSynthesis filter (upsampling):" << std::endl;
    std::cout << "  Sum: " << p_sum << " (should be ~2.0)" << std::endl;
    std::cout << "  Sum of squares: " << p_sum_sq << std::endl;
    
    // Check if this is a valid quadratic B-spline refinement
    std::cout << "\nQuadratic B-spline refinement coefficients:" << std::endl;
    std::cout << "  p[0] = " << pCoeffs[0] << " (should be 0.25)" << std::endl;
    std::cout << "  p[1] = " << pCoeffs[1] << " (should be 0.75)" << std::endl;
    std::cout << "  p[2] = " << pCoeffs[2] << " (should be 0.75)" << std::endl;
    std::cout << "  p[3] = " << pCoeffs[3] << " (should be 0.25)" << std::endl;
    
    // Test perfect reconstruction property
    std::cout << "\n=== PERFECT RECONSTRUCTION TEST ===" << std::endl;
    std::cout << "For perfect reconstruction, we need:" << std::endl;
    std::cout << "1. Analysis filter preserves DC (sum ≈ √2)" << std::endl;
    std::cout << "2. Synthesis filter doubles samples (sum = 2)" << std::endl;
    std::cout << "3. Combined system has gain of √2 * 2 / 2 = √2" << std::endl;
    
    float combined_gain = a_sum * p_sum / 2.0f; // The /2 is from decimation
    std::cout << "\nCombined gain: " << combined_gain << " (should be ~√2 = 1.414)" << std::endl;
    
    // The issue might be normalization
    std::cout << "\n=== POTENTIAL ISSUE ===" << std::endl;
    if (std::abs(combined_gain - std::sqrt(2.0f)) > 0.1f) {
        std::cout << "WARNING: Combined gain is not √2!" << std::endl;
        std::cout << "This will cause energy loss in the low-pass branch." << std::endl;
        
        float correction_factor = std::sqrt(2.0f) / combined_gain;
        std::cout << "\nTo fix, multiply analysis coefficients by: " << correction_factor << std::endl;
    }
}

int main() {
    analyzeFilterCoefficients();
    return 0;
}