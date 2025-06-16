#include "FFT.hpp"
#include <cmath>    // For M_PI
#include <algorithm> // for std::swap
#include <iostream>  // For std::cerr

namespace FFT {

// Helper for bit reversal
void bitReversalPermutation(std::vector<Complex>& data) {
    int n = data.size();
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j ^= bit;
        if (i < j)
            std::swap(data[i], data[j]);
    }
}

void fft1D(std::vector<Complex>& data, bool inverse) {
    int n = data.size();
    if (n == 0 || (n & (n - 1)) != 0) { // Check if n is a power of 2
        // For simplicity, we'll just return or throw. Production code might pad.
        // std::cerr << "FFT data size must be a power of 2. Size: " << n << std::endl;
        return; 
    }

    bitReversalPermutation(data);

    for (int len = 2; len <= n; len <<= 1) {
        float angle = 2 * M_PI / len * (inverse ? -1 : 1);
        Complex wlen(std::cos(angle), std::sin(angle));
        for (int i = 0; i < n; i += len) {
            Complex w(1);
            for (int j = 0; j < len / 2; j++) {
                Complex u = data[i + j];
                Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }

    if (inverse) {
        for (Complex& val : data) {
            val /= n;
        }
    }
}


void fft2D(Image& real_part, Image& imag_part, bool inverse) {
    int width = real_part.width;
    int height = real_part.height;

    if (width != imag_part.width || height != imag_part.height) {
        // Error: Mismatched dimensions
        return;
    }
    // Check if width and height are powers of 2 (optional, but fft1D assumes it)
    if ((width & (width - 1)) != 0 || (height & (height - 1)) != 0) {
        std::cerr << "Warning: FFT2D dimensions are not powers of 2. FFT might not work correctly." << std::endl;
    }


    // FFT rows
    std::vector<Complex> row_data(width);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            row_data[x] = Complex(real_part.at(x, y), imag_part.at(x, y));
        }
        fft1D(row_data, inverse);
        for (int x = 0; x < width; ++x) {
            real_part.at(x, y) = row_data[x].real();
            imag_part.at(x, y) = row_data[x].imag();
        }
    }

    // FFT columns
    std::vector<Complex> col_data(height);
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            col_data[y] = Complex(real_part.at(x, y), imag_part.at(x, y));
        }
        fft1D(col_data, inverse);
        for (int y = 0; y < height; ++y) {
            real_part.at(x, y) = col_data[y].real();
            imag_part.at(x, y) = col_data[y].imag();
        }
    }
}


void fftShift(Image& image_real, Image& image_imag) {
    int w = image_real.width;
    int h = image_real.height;
    int w_half = w / 2;
    int h_half = h / 2;

    Image temp_real(w, h);
    Image temp_imag(w, h);

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int new_x = (x + w_half) % w;
            int new_y = (y + h_half) % h;
            temp_real.at(new_x, new_y) = image_real.at(x, y);
            temp_imag.at(new_x, new_y) = image_imag.at(x, y);
        }
    }
    image_real = temp_real;
    image_imag = temp_imag;
}

} // namespace FFT