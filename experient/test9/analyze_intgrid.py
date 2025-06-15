import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import os

# (The function 'load_raw_data' is the same as before)
def load_raw_data(filename, size=256):
    try:
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
        expected_size = size * size
        if data.size != expected_size: return None
        return data.reshape((size, size))
    except FileNotFoundError:
        return None

# (The function 'plot_noise_column' is the same as before)
def plot_noise_column(ax_col, data, title):
    if data is None:
        for ax in ax_col:
            ax.text(0.5, 0.5, 'Data not found', ha='center', va='center', color='red')
            ax.axis('off')
        ax_col[0].set_title(title, fontsize=12)
        return

    data_centered = data - np.mean(data)
    F = fftshift(fft2(data_centered))
    power_spectrum = np.abs(F)**2

    h, w = power_spectrum.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx)**2 + (Y - cy)**2)
    
    max_r = min(cx, cy)
    radial_profile = [np.mean(power_spectrum[(R >= r) & (R < r + 1)]) if np.any((R >= r) & (R < r + 1)) else 0 for r in range(max_r)]
    freqs = np.arange(len(radial_profile)) / (2 * max_r)
    
    # Plotting...
    ax = ax_col[0]
    ax.imshow(data, cmap='gray')
    ax.set_title(title, fontsize=11, pad=8)
    ax.axis('off')

    ax = ax_col[1]
    power_log = np.log1p(power_spectrum.copy())
    power_log[cy, cx] = 0
    vmax = np.percentile(power_log, 99.8)
    ax.imshow(power_log, cmap='hot', vmin=0, vmax=vmax)
    ax.axis('off')

    ax = ax_col[2]
    ax.semilogy(freqs, radial_profile, linewidth=1.5)
    ax.set_xlim(0, 0.5)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency', fontsize=9)
    peak_freq_idx = np.argmax(radial_profile)
    # Don't show peak line for sliced noise as it's at f=0
    if peak_freq_idx > 2 and 'Sliced' not in title:
        peak_freq = freqs[peak_freq_idx]
        ax.axvline(peak_freq, color='r', linestyle='--', linewidth=1, label=f'Peak~{peak_freq:.3f}')
        ax.legend(fontsize=7)

    if ax.get_subplotspec().colspan.start == 0:
        ax_col[0].set_ylabel('Noise Pattern', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[1].set_ylabel('Power Spectrum', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[2].set_ylabel('Radial Profile', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)


def create_intgrid_comparison_figure():
    """Generates the comparison figure for the integer grid sampling test."""
    
    file_2d = "wavelet_noise_2D_octave_intgrid.raw"
    file_3d_sliced = "wavelet_noise_3Dsliced_octave_intgrid.raw"
    file_3d_projected = "wavelet_noise_3Dprojected_octave_intgrid.raw"
    
    data_2d = load_raw_data(file_2d)
    data_3d_s = load_raw_data(file_3d_sliced)
    data_3d_p = load_raw_data(file_3d_projected)
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    
    fig.suptitle('Wavelet Noise Analysis - Integer Grid Sampling\n(Replicating Ideal Conditions)', fontsize=18, y=1.05)

    plot_noise_column(axes[:, 0], data_2d, "Pure 2D Noise")
    plot_noise_column(axes[:, 1], data_3d_s, "3D Sliced Noise")
    plot_noise_column(axes[:, 2], data_3d_p, "3D Projected Noise (IDEAL)")

    for ax in axes[:, 0]: ax.spines['left'].set_color('green'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 1]: ax.spines['left'].set_color('red'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 2]: ax.spines['left'].set_color('blue'); ax.spines['left'].set_linewidth(4)

    output_filename = "wavelet_intgrid_comparison.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nInteger grid comparison figure saved to '{output_filename}'")
    plt.show()

if __name__ == "__main__":
    create_intgrid_comparison_figure()