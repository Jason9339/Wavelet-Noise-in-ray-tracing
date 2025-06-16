#ifndef FFT_HPP
#define FFT_HPP

#include "Image.hpp"
#include <vector>
#include <complex>

namespace FFT {
    using Complex = std::complex<float>;

    // Performs 1D FFT (Cooley-Tukey Radix-2)
    // Data size must be a power of 2
    void fft1D(std::vector<Complex>& data, bool inverse);

    // Performs 2D FFT on real-valued image data
    // Output is in `real_part` and `imag_part`
    void fft2D(Image& real_part, Image& imag_part, bool inverse);

    // Shifts the DC component of FFT to the center for visualization
    void fftShift(Image& image_real, Image& image_imag);
}

#endif // FFT_HPP