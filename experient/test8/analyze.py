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
        print(f"Warning: File not found - {filename}")
        return None

def plot_noise_column(ax_col, data, title):
    """Plots a full analysis column (pattern, fft, radial) for given data."""
    if data is None:
        for ax in ax_col:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center')
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
    radial_profile = [np.mean(power_spectrum[(R >= r) & (R < r + 1)]) for r in range(max_r)]
    freqs = np.arange(len(radial_profile)) / (2 * max_r)
    
    # --- Plotting ---
    # 1. Noise Pattern
    ax = ax_col[0]
    im = ax.imshow(data, cmap='gray')
    ax.set_title(title, fontsize=12, pad=10)
    ax.axis('off')

    # 2. Power Spectrum (log scale)
    ax = ax_col[1]
    # Set DC component to zero to improve contrast of the bands
    power_log = np.log1p(power_spectrum.copy())
    power_log[cy, cx] = 0
    vmax = np.percentile(power_log, 99.8) # Avoid extreme bright spots
    ax.imshow(power_log, cmap='hot', vmin=0, vmax=vmax)
    ax.axis('off')

    # 3. Radial Profile
    ax = ax_col[2]
    ax.semilogy(freqs, radial_profile, linewidth=1.5)
    ax.set_xlim(0, 0.5)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency')
    peak_freq_idx = np.argmax(radial_profile)
    # Avoid labeling peak at DC for sliced noise
    if peak_freq_idx > 2:
        peak_freq = freqs[peak_freq_idx]
        ax.axvline(peak_freq, color='r', linestyle='--', linewidth=1, label=f'Peak~{peak_freq:.3f}')
        ax.legend(fontsize=8)
    
    if ax.get_subplotspec().colspan.start == 0:
        ax.set_ylabel('Power')

def create_comparison_figure(two_d_files, three_d_files):
    """Generates the main comparison figure for 2D vs 3D sliced noise."""
    num_octaves = len(two_d_files)
    num_cols = num_octaves * 2 + 1 # 3 for 2D, 1 for space, 3 for 3D
    num_rows = 3
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 9),
                             gridspec_kw={'width_ratios': [1,1,1, 0.2, 1,1,1], 'wspace': 0.6, 'hspace': 0.4})
    
    fig.suptitle('Wavelet Noise: Pure 2D vs. 3D Sliced (Reproducing Figure 8 concepts)', fontsize=18, y=0.99)
    
    # Set labels for rows
    axes[0, 0].set_ylabel('Noise Pattern', fontsize=12, labelpad=20)
    axes[1, 0].set_ylabel('Power Spectrum (FFT)', fontsize=12, labelpad=20)
    axes[2, 0].set_ylabel('Radial Power Profile', fontsize=12, labelpad=20)

    # --- Pure 2D Noise ---
    for i, filename in enumerate(two_d_files):
        data = load_raw_data(filename)
        octave = filename.split('_')[-1].split('.')[0]
        title = f"Pure 2D - Octave {octave}"
        plot_noise_column(axes[:, i], data, title)

    # --- Separator ---
    for row in range(num_rows):
        axes[row, num_octaves].axis('off')

    # --- 3D Sliced Noise ---
    for i, filename in enumerate(three_d_files):
        col_idx = i + num_octaves + 1
        data = load_raw_data(filename)
        octave = filename.split('_')[-1].split('.')[0]
        title = f"3D Sliced - Octave {octave}"
        plot_noise_column(axes[:, col_idx], data, title)
        
    fig.text(0.27, 0.92, '✓ Truly Band-Limited', ha='center', va='center', fontsize=14, weight='bold', color='green')
    fig.text(0.73, 0.92, '✗ Low-Frequency Leakage', ha='center', va='center', fontsize=14, weight='bold', color='red')

    plt.tight_layout(rect=[0.03, 0, 1, 0.9])
    
    output_filename = "wavelet_2D_vs_3D_sliced_comparison.png"
    plt.savefig(output_filename, dpi=150)
    print(f"\nComparison figure saved to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    two_d_files = [f"wavelet_noise_2D_octave_{oct}.raw" for oct in [3, 4, 5]]
    three_d_files = [f"wavelet_noise_3Dsliced_octave_{oct}.raw" for oct in [3, 4, 5]]
    
    # Check if files exist
    all_files_exist = all(os.path.exists(f) for f in two_d_files + three_d_files)
    
    if not all_files_exist:
        print("Error: Not all required .raw files were found.")
        print("Please compile and run the C++ program first.")
    else:
        print("Analyzing generated noise files...")
        create_comparison_figure(two_d_files, three_d_files)