#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include "WaveletNoise.h"

void save2DSlice(const std::string& filename, const WaveletNoise& wn, int pz, int vizSize) {
    std::ofstream out(filename, std::ios::binary);
    float p[3];
    for (int y = 0; y < vizSize; ++y) {
        for (int x = 0; x < vizSize; ++x) {
            p[0] = static_cast<float>(x) / vizSize;
            p[1] = static_cast<float>(y) / vizSize;
            p[2] = static_cast<float>(pz) / wn.getTileSize();
            float val = wn.evaluate3D(p);
            out.write(reinterpret_cast<const char*>(&val), sizeof(float));
        }
    }
    out.close();
}

int main() {
    std::vector<int> tileSizes = {128};
    std::vector<unsigned int> seeds = {1};
    std::vector<int> pz_slices = {16};
    const int vizSize = 256;

    for (int tile : tileSizes) {
        for (unsigned int seed : seeds) {
            WaveletNoise wn(tile, seed);
            wn.generateNoiseTile();
            for (int pz : pz_slices) {
                std::string filename = "wavelet_noise_output_tilesize_" + std::to_string(tile)
                                     + "_seed_" + std::to_string(seed)
                                     + "_pz_" + std::to_string(pz)
                                     + ".raw";
                std::cout << "Generating: " << filename << std::endl;
                save2DSlice(filename, wn, pz, vizSize);
            }
        }
    }

    std::cout << "All tiles completed." << std::endl;
    return 0;
}
