#include "Image.hpp"
#include "PerlinNoise.hpp"
#include "WaveletNoise.hpp"
#include "FFT.hpp"
#include <iostream>
#include <string>
#include <vector> // For std::vector
#include <cmath>  // For std::log, std::fabs
#include <functional> // For std::function

void generate_and_save_noise_and_spectrum(
    const std::string& name_prefix,
    int size,
    const std::function<float(float, float)>& noise_func,
    float scale_factor // To control feature size in noise
) {
    Image noise_image(size, size);
    Image noise_image_real_fft(size, size); // For FFT input/output
    Image noise_image_imag_fft(size, size); // For FFT input/output (initially zero)

    std::cout << "Generating " << name_prefix << " noise (" << size << "x" << size << ")..." << std::endl;
    float min_n = 1e10, max_n = -1e10;
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            // Scale coordinates to control feature size.
            // Larger scale_factor = smaller features (higher frequency).
            float val = noise_func(static_cast<float>(x) / size * scale_factor,
                                   static_cast<float>(y) / size * scale_factor);
            if(val < min_n) min_n = val;
            if(val > max_n) max_n = val;
            noise_image.at(x, y) = val;
            noise_image_real_fft.at(x,y) = val; // Prepare for FFT
            noise_image_imag_fft.at(x,y) = 0.0f; 
        }
    }
    std::cout << name_prefix << " Noise raw range: [" << min_n << ", " << max_n << "]" << std::endl;
    noise_image.savePPM(name_prefix + "_noise.ppm", false, true); // Normalize for saving

    std::cout << "Computing " << name_prefix << " spectrum..." << std::endl;
    // Optional: Multiply by (-1)^(x+y) before FFT to center DC
    for(int y=0; y<size; ++y) {
        for(int x=0; x<size; ++x) {
            if((x+y)%2 != 0) {
                noise_image_real_fft.at(x,y) *= -1.0f;
            }
        }
    }

    FFT::fft2D(noise_image_real_fft, noise_image_imag_fft, false);
    // FFT::fftShift(noise_image_real_fft, noise_image_imag_fft); // Alternative way to center DC

    Image spectrum_image(size, size);
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            float re = noise_image_real_fft.at(x, y);
            float im = noise_image_imag_fft.at(x, y);
            spectrum_image.at(x, y) = re * re + im * im; // Power = magnitude squared
        }
    }
    // Power spectrum is often viewed on a log scale and normalized
    spectrum_image.savePPM(name_prefix + "_spectrum.ppm", true, true);
    std::cout << name_prefix << " processing complete." << std::endl << std::endl;
}


int main() {
    const int IMAGE_SIZE = 256; // Must be power of 2 for this simple FFT
    const int WAVELET_TILE_SIZE = 64; // Must be power of 2

    // --- Perlin Noise ---
    PerlinNoise perlin(123); // Seed for reproducibility
    generate_and_save_noise_and_spectrum(
        "perlin",
        IMAGE_SIZE,
        [&](float x, float y){ return perlin.noise(x, y); },
        8.0f // Scale factor for Perlin (e.g., 4-16 octaves visually)
    );

    // --- Wavelet Noise ---
    try {
        WaveletNoise wavelet(WAVELET_TILE_SIZE, 456); // Seed
        generate_and_save_noise_and_spectrum(
            "wavelet",
            IMAGE_SIZE,
            [&](float x_norm, float y_norm){ // x_norm, y_norm are [0, scale_factor]
                // Wavelet noise function expects coordinates in tile units
                // If x_norm is [0, S], and tile is size T, then x_tile = x_norm * T / S
                // but here, wavelet.noise takes x in [0,1) range relative to tile.
                // so if x_norm from [0, scale_factor], then x_tile_relative = fmod(x_norm, 1.0)
                // We want to sample over 'scale_factor' number of tiles.
                return wavelet.noise(x_norm, y_norm); 
            },
            static_cast<float>(IMAGE_SIZE) / static_cast<float>(WAVELET_TILE_SIZE) * 2.0f // e.g. cover 2x2 tiles over image
            // A scale_factor of 4.0 means the IMAGE_SIZE will span 4 wavelet tiles.
        );
    } catch (const std::exception& e) {
        std::cerr << "Error with Wavelet Noise: " << e.what() << std::endl;
        return 1;
    }


    std::cout << "All done. Check PPM files." << std::endl;
    return 0;
}