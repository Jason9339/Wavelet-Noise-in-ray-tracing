import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import os

def load_raw_data(filename, size=256):
    """Loads a raw float32 file into a numpy array."""
    try:
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        expected_size = size * size
        if data.size != expected_size:
            print(f"Warning: Size mismatch in {filename}. Expected {expected_size}, got {data.size}.")
            return None
        return data.reshape((size, size))
    except FileNotFoundError:
        print(f"Error: File not found - {filename}")
        return None

def plot_noise_column(ax_col, data, title):
    """Plots a full analysis column (pattern, fft, radial) for given data."""
    if data is None:
        for ax in ax_col:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', color='red')
            ax.axis('off')
        ax_col[0].set_title(title, fontsize=12)
        return

    # --- Calculations ---
    data_centered = data - np.mean(data)
    F = fftshift(fft2(data_centered))
    power_spectrum = np.abs(F) ** 2

    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    max_r = min(cx, cy)
    # Handle potential empty masks
    radial_profile = [np.mean(power_spectrum[(R >= r) & (R < r + 1)]) if np.any((R >= r) & (R < r + 1)) else 0 for r in range(max_r)]

    freqs = np.arange(len(radial_profile)) / (2 * max_r)
    
    # --- Plotting ---
    # 1. Noise Pattern
    ax = ax_col[0]
    ax.imshow(data, cmap='gray')
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis('off')

    # 2. Power Spectrum (log scale)
    ax = ax_col[1]
    power_log = np.log1p(power_spectrum.copy())
    power_log[cy, cx] = 0
    vmax = np.percentile(power_log, 99.8)
    ax.imshow(power_log, cmap='hot', vmin=0, vmax=vmax)
    ax.axis('off')

    # 3. Radial Profile
    ax = ax_col[2]
    ax.semilogy(freqs, radial_profile, linewidth=1.5)
    ax.set_xlim(0, 0.5)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency', fontsize=9)
    peak_freq_idx = np.argmax(radial_profile)
    if peak_freq_idx > 2:
        peak_freq = freqs[peak_freq_idx]
        ax.axvline(peak_freq, color='r', linestyle='--', linewidth=1, label=f'Peak~{peak_freq:.3f}')
        ax.legend(fontsize=7)
    
    # Set y-label only for the first column of each type
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel('Noise\nPattern', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
    if ax.get_subplotspec().colspan.start == 0:
        ax_col[1].set_ylabel('Power\nSpectrum', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[2].set_ylabel('Radial\nProfile', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)

def create_full_comparison_figure(octave):
    """Generates the main comparison figure for a single octave."""
    
    file_2d = f"wavelet_noise_2D_octave_{octave}.raw"
    file_3d_sliced = f"wavelet_noise_3Dsliced_octave_{octave}.raw"
    file_3d_projected = f"wavelet_noise_3Dprojected_octave_{octave}.raw"
    
    data_2d = load_raw_data(file_2d)
    data_3d_s = load_raw_data(file_3d_sliced)
    data_3d_p = load_raw_data(file_3d_projected)
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    
    fig.suptitle(f'Wavelet Noise Analysis - Octave {octave}\n(Reproducing Figure 8 concepts)', fontsize=18, y=1.05)

    # --- Column 1: Pure 2D Noise ---
    plot_noise_column(axes[:, 0], data_2d, "Pure 2D Noise\n(Truly Band-Limited)")
    
    # --- Column 2: 3D Sliced Noise ---
    plot_noise_column(axes[:, 1], data_3d_s, "3D Sliced Noise\n(Low-Frequency Leakage)")

    # --- Column 3: 3D Projected Noise ---
    plot_noise_column(axes[:, 2], data_3d_p, "3D Projected Noise\n(Band-Limiting Preserved)")

    # Add color indicators for clarity
    for ax in axes[:, 0]: ax.spines['left'].set_color('green'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 1]: ax.spines['left'].set_color('red'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 2]: ax.spines['left'].set_color('blue'); ax.spines['left'].set_linewidth(4)

    output_filename = f"wavelet_full_comparison_octave_{octave}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nFull comparison figure for octave {octave} saved to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    
    octaves_to_analyze = [3, 4, 5]
    all_files_exist = True
    for oct in octaves_to_analyze:
        files = [
            f"wavelet_noise_2D_octave_{oct}.raw",
            f"wavelet_noise_3Dsliced_octave_{oct}.raw",
            f"wavelet_noise_3Dprojected_octave_{oct}.raw"
        ]
        if not all(os.path.exists(f) for f in files):
            all_files_exist = False
            print(f"Error: Missing files for octave {oct}. Please run C++ generator.")
            break

    if all_files_exist:
        print("All required .raw files found. Starting analysis...")
        for oct in octaves_to_analyze:
            create_full_comparison_figure(oct)
    else:
        print("\nAborting analysis due to missing files.")