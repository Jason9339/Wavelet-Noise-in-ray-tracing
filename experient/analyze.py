import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift
import os

def load_raw_data(filename, size=256):
    """Loads a raw float32 file into a numpy array."""
    try:
        # Add result_raw prefix if not already present
        if not filename.startswith('result_raw/'):
            filename = f'result_raw/{filename}'
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
    # Don't show peak line for sliced noise as it's at f=0, and avoid DC component
    if peak_freq_idx > 2 and 'Sliced' not in title:
        peak_freq = freqs[peak_freq_idx]
        ax.axvline(peak_freq, color='r', linestyle='--', linewidth=1, label=f'Peak~{peak_freq:.3f}')
        ax.legend(fontsize=7)
    
    if ax.get_subplotspec().colspan.start == 0:
        ax_col[0].set_ylabel('Noise\nPattern', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[1].set_ylabel('Power\nSpectrum', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[2].set_ylabel('Radial\nProfile', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)

def analyze_single_band(filename, title):
    """Analyze a single band noise pattern to show band-limited property (detailed 5-column layout)"""
    data = load_raw_data(filename)
    if data is None:
        print(f"Could not read {filename}")
        return None, None

    data_centered = data - np.mean(data)
    F = fftshift(fft2(data_centered))
    magnitude = np.abs(F)
    power_spectrum = magnitude ** 2

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(f'{title}\n(Raw Values - Detailed Band Analysis)', fontsize=14, y=1.08)

    # 1. Noise Pattern
    ax = axes[0]
    ax.imshow(data, cmap='gray')
    ax.set_title("Noise Pattern")
    ax.axis('off')

    # 2. FFT Magnitude (log scale)
    ax = axes[1]
    magnitude_log = np.log1p(magnitude)
    ax.imshow(magnitude_log, cmap='gray')
    ax.set_title("FFT Magnitude (log)")
    ax.axis('off')

    # 3. Power Spectrum (log scale)
    ax = axes[2]
    power_log = np.log1p(power_spectrum)
    ax.imshow(power_log, cmap='gray')
    ax.set_title("Power Spectrum (log)")
    ax.axis('off')

    # 4. Band Structure with peak radius detection
    ax = axes[3]
    power_norm = power_spectrum / np.max(power_spectrum)
    im = ax.imshow(power_norm, cmap='gray', vmin=0, vmax=1e-2)

    h, w = power_norm.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)

    # Calculate radial profile for peak detection
    max_r = min(cx, cy)
    radial_profile = []
    for r in range(max_r):
        mask = (R >= r) & (R < r + 1)
        if np.any(mask):
            radial_profile.append(np.mean(power_spectrum[mask]))
        else:
            radial_profile.append(0)

    peak_radius = np.argmax(radial_profile)
    ax.set_title(f"Band Structure\n(peak at r≈{peak_radius})")
    ax.axis('off')

    # 5. Radial Power Spectrum
    ax = axes[4]
    radii = np.arange(len(radial_profile))
    normalized_freq = radii / (2 * len(radial_profile))

    ax.semilogy(normalized_freq, radial_profile, 'b-', linewidth=2)
    if peak_radius > 2:  # Avoid DC component
        ax.axvline(normalized_freq[peak_radius], color='r', linestyle='--', 
                   label=f'Peak at f≈{normalized_freq[peak_radius]:.3f}')
        ax.legend(fontsize=8)
    ax.set_xlabel('Normalized Frequency')
    ax.set_ylabel('Power')
    ax.set_title('Radial Power Spectrum')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.5)

    plt.tight_layout()
    return fig, peak_radius

def create_detailed_individual_analysis():
    """Generate detailed 5-column analysis for all available files"""
    print("\nGenerating Detailed Individual Analysis (5-Column Layout)...")
    
    # All possible files to analyze
    all_files = []
    
    # Integer grid files
    intgrid_files = [
        ("wavelet_noise_2D_octave_intgrid.raw", "2D Integer Grid"),
        ("wavelet_noise_3Dsliced_octave_intgrid.raw", "3D Sliced Integer Grid"),
        ("wavelet_noise_3Dprojected_octave_intgrid.raw", "3D Projected Integer Grid")
    ]
    
    # Standard octave files
    for octave in [3, 4, 5]:
        octave_files = [
            (f"wavelet_noise_2D_octave_{octave}.raw", f"2D Octave {octave}"),
            (f"wavelet_noise_3Dsliced_octave_{octave}.raw", f"3D Sliced Octave {octave}"),
            (f"wavelet_noise_3Dprojected_octave_{octave}.raw", f"3D Projected Octave {octave}")
        ]
        all_files.extend(octave_files)
    
    all_files.extend(intgrid_files)
    
    # Generate detailed analysis for each existing file
    for filename, description in all_files:
        # Check if file exists in result_raw directory
        file_path = f'result_raw/{filename}' if not filename.startswith('result_raw/') else filename
        if os.path.exists(file_path):
            print(f"   Analyzing: {description}")
            fig, peak_r = analyze_single_band(filename, description)
            if fig:
                output_name = f'result_analyze/{filename.replace(".raw", "")}_detailed_analysis.png'
                plt.savefig(output_name, dpi=150, bbox_inches='tight')
                print(f"   → Saved: {output_name}")
                plt.show()
        else:
            print(f"   Warning: Skipping {description}: file not found")

def create_intgrid_comparison_figure():
    """Generates the comparison figure for the integer grid sampling test."""
    
    file_2d = "wavelet_noise_2D_octave_intgrid.raw"
    file_3d_sliced = "wavelet_noise_3Dsliced_octave_intgrid.raw"
    file_3d_projected = "wavelet_noise_3Dprojected_octave_intgrid.raw"
    
    data_2d = load_raw_data(file_2d)
    data_3d_s = load_raw_data(file_3d_sliced)
    data_3d_p = load_raw_data(file_3d_projected)
    
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), constrained_layout=True)
    
    fig.suptitle('Wavelet Noise Analysis - Integer Grid Sampling\n(Raw Values - No [0,1] Normalization)', fontsize=18, y=1.05)

    plot_noise_column(axes[:, 0], data_2d, "Pure 2D Noise\n(Integer Grid)")
    plot_noise_column(axes[:, 1], data_3d_s, "3D Sliced Noise\n(Low-Freq Leakage)")
    plot_noise_column(axes[:, 2], data_3d_p, "3D Projected Noise\n(Ideal Band-Limiting)")

    # Add color indicators for clarity
    for ax in axes[:, 0]: ax.spines['left'].set_color('green'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 1]: ax.spines['left'].set_color('red'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 2]: ax.spines['left'].set_color('blue'); ax.spines['left'].set_linewidth(4)

    output_filename = "result_analyze/wavelet_intgrid_comparison.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nInteger grid comparison figure saved to '{output_filename}'")
    plt.show()

def plot_noise_column_extended(ax_col, data, title):
    """Plots a full analysis column (pattern, fft, radial, normalized power) for given data."""
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

    # 3. Normalized Power Spectrum (gray scale with new vmax)
    ax = ax_col[2]
    power_norm = power_spectrum / np.max(power_spectrum)
    ax.imshow(power_norm, cmap='gray', vmin=0, vmax=1e-2)
    ax.axis('off')

    # 4. Radial Profile
    ax = ax_col[3]
    ax.semilogy(freqs, radial_profile, linewidth=1.5)
    ax.set_xlim(0, 0.5)
    ax.grid(True, which="both", linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency', fontsize=9)
    peak_freq_idx = np.argmax(radial_profile)
    # Don't show peak line for sliced noise as it's at f=0, and avoid DC component
    if peak_freq_idx > 2 and 'Sliced' not in title:
        peak_freq = freqs[peak_freq_idx]
        ax.axvline(peak_freq, color='r', linestyle='--', linewidth=1, label=f'Peak~{peak_freq:.3f}')
        ax.legend(fontsize=7)
    
    # Set y-label only for the first column of each type
    if ax.get_subplotspec().colspan.start == 0:
        ax_col[0].set_ylabel('Noise\nPattern', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[1].set_ylabel('Power\nSpectrum\n(log)', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[2].set_ylabel('Normalized\nPower\n(gray)', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)
        ax_col[3].set_ylabel('Radial\nProfile', fontsize=9, rotation=0, ha='right', va='center', labelpad=25)

def create_octave_comparison_figure(octave):
    """Generates the main comparison figure for a single octave."""
    
    file_2d = f"wavelet_noise_2D_octave_{octave}.raw"
    file_3d_sliced = f"wavelet_noise_3Dsliced_octave_{octave}.raw"
    file_3d_projected = f"wavelet_noise_3Dprojected_octave_{octave}.raw"
    
    data_2d = load_raw_data(file_2d)
    data_3d_s = load_raw_data(file_3d_sliced)
    data_3d_p = load_raw_data(file_3d_projected)
    
    fig, axes = plt.subplots(4, 3, figsize=(14, 12), constrained_layout=True)
    
    fig.suptitle(f'Wavelet Noise Analysis - Octave {octave}', fontsize=18, y=1.02)

    # --- Column 1: Pure 2D Noise ---
    plot_noise_column_extended(axes[:, 0], data_2d, "Pure 2D Noise\n(Truly Band-Limited)")
    
    # --- Column 2: 3D Sliced Noise ---
    plot_noise_column_extended(axes[:, 1], data_3d_s, "3D Sliced Noise\n(Low-Frequency Leakage)")

    # --- Column 3: 3D Projected Noise ---
    plot_noise_column_extended(axes[:, 2], data_3d_p, "3D Projected Noise\n(Band-Limiting Preserved)")

    # Add color indicators for clarity
    for ax in axes[:, 0]: ax.spines['left'].set_color('green'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 1]: ax.spines['left'].set_color('red'); ax.spines['left'].set_linewidth(4)
    for ax in axes[:, 2]: ax.spines['left'].set_color('blue'); ax.spines['left'].set_linewidth(4)

    output_filename = f"result_analyze/wavelet_full_comparison_octave_{octave}.png"
    plt.savefig(output_filename, dpi=150, bbox_inches='tight')
    print(f"\nFull comparison figure for octave {octave} saved to '{output_filename}'")
    plt.show()

def print_data_statistics():
    """Print statistical information about all generated data files."""
    print("\n" + "="*80)
    print("DATA STATISTICS SUMMARY (Raw Values - No [0,1] Normalization)")
    print("="*80)
    
    # Integer grid files
    intgrid_files = [
        ("wavelet_noise_2D_octave_intgrid.raw", "2D Integer Grid"),
        ("wavelet_noise_3Dsliced_octave_intgrid.raw", "3D Sliced Integer Grid"),
        ("wavelet_noise_3Dprojected_octave_intgrid.raw", "3D Projected Integer Grid")
    ]
    
    print("\n--- Integer Grid Sampling Results ---")
    for filename, description in intgrid_files:
        data = load_raw_data(filename)
        if data is not None:
            print(f"{description:25s}: mean={np.mean(data):8.4f}, std={np.std(data):8.4f}, min={np.min(data):8.4f}, max={np.max(data):8.4f}")
        else:
            print(f"{description:25s}: FILE NOT FOUND")
    
    # Standard octave files
    octaves = [3, 4, 5]
    print("\n--- Standard Octave Results ---")
    for octave in octaves:
        print(f"\nOctave {octave}:")
        octave_files = [
            (f"wavelet_noise_2D_octave_{octave}.raw", f"  2D Octave {octave}"),
            (f"wavelet_noise_3Dsliced_octave_{octave}.raw", f"  3D Sliced Octave {octave}"),
            (f"wavelet_noise_3Dprojected_octave_{octave}.raw", f"  3D Projected Octave {octave}")
        ]
        
        for filename, description in octave_files:
            data = load_raw_data(filename)
            if data is not None:
                print(f"{description:25s}: mean={np.mean(data):8.4f}, std={np.std(data):8.4f}, min={np.min(data):8.4f}, max={np.max(data):8.4f}")
            else:
                print(f"{description:25s}: FILE NOT FOUND")

def get_power_spectrum(data):
    """Computes the log-scaled power spectrum of a 2D array."""
    if data is None:
        return None
    data_centered = data - np.mean(data)
    F = fftshift(fft2(data_centered))
    power_spectrum = np.abs(F) ** 2
    return np.log1p(power_spectrum)

def plot_pattern_and_fft(ax_pattern, ax_fft, data, title):
    """Helper function to plot a noise pattern and its FFT side-by-side."""
    if data is None:
        ax_pattern.text(0.5, 0.5, 'Data\nNot Found', ha='center', va='center', color='red')
        ax_pattern.axis('off')
        ax_fft.text(0.5, 0.5, 'FFT\nNot Found', ha='center', va='center', color='red')
        ax_fft.axis('off')
        ax_pattern.set_title(title, fontsize=10)
        return

    power_spectrum_log = get_power_spectrum(data)
    
    ax_pattern.imshow(data, cmap='gray')
    ax_pattern.set_title(title, fontsize=10)
    ax_pattern.axis('off')

    # 顯示功率譜
    vmax = np.percentile(power_spectrum_log, 99.9) # 調整對比度
    ax_fft.imshow(power_spectrum_log, cmap='hot', vmin=0, vmax=vmax)
    ax_fft.set_title("Fourier Transform", fontsize=10)
    ax_fft.axis('off')

def create_figure8_comparison(octave=4):
    """
    Recreates the layout and spirit of Figure 8 from the paper,
    comparing Perlin and Wavelet noise side-by-side.
    """
    print(f"\nGenerating Figure 8 comparison for Octave {octave}...")
    o_str = str(octave)
    
    p_2d = load_raw_data(f"perlin_noise_2D_octave_{o_str}.raw")
    p_3d_slice = load_raw_data(f"perlin_noise_3Dsliced_octave_{o_str}.raw")
    
    w_2d = load_raw_data(f"wavelet_noise_2D_octave_{o_str}.raw")
    w_3d_slice = load_raw_data(f"wavelet_noise_3Dsliced_octave_{o_str}.raw")
    w_3d_proj = load_raw_data(f"wavelet_noise_3Dprojected_octave_{o_str}.raw")

    if any(d is None for d in [p_2d, p_3d_slice, w_2d, w_3d_slice, w_3d_proj]):
        print("Warning: Missing one or more data files for Figure 8 comparison. Skipping.")
        return

    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    fig.suptitle(f'Figure 8 Replication: Perlin vs. Wavelet Noise (Octave {octave})', fontsize=16, y=0.98)

    axes[2, 0].remove()
    axes[2, 1].remove()

    plot_pattern_and_fft(axes[0, 0], axes[0, 1], p_2d, "(a) 2D Perlin noise")
    plot_pattern_and_fft(axes[1, 0], axes[1, 1], p_3d_slice, "(b) 2D slice through 3D Perlin noise")
    plot_pattern_and_fft(axes[0, 2], axes[0, 3], w_2d, "(d) 2D wavelet noise")
    plot_pattern_and_fft(axes[1, 2], axes[1, 3], w_3d_slice, "(e) 2D slice through 3D wavelet noise")
    plot_pattern_and_fft(axes[2, 2], axes[2, 3], w_3d_proj, "(f) 3D wavelet noise projected onto 2D")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_filename = f"result_analyze/figure8_comparison_octave_{o_str}.png"
    plt.savefig(output_filename, dpi=150)
    print(f"   → Saved: {output_filename}")
    plt.show()

def create_figure9_comparison(octaves=[3, 4, 5]):
    """
    Recreates the spirit of Figure 9 from the paper by visualizing
    the separation of frequency bands.
    """
    print(f"\nGenerating Figure 9 comparison for Octaves {octaves}...")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Figure 9 Replication: Frequency Band Separation', fontsize=16, y=1.02)
    
    # --- Perlin Noise ---
    perlin_bands = []
    for o in octaves:
        data = load_raw_data(f"perlin_noise_2D_octave_{o}.raw")
        if data is None: 
            print("Warning: Missing Perlin data for Figure 9. Skipping Perlin plot.")
            perlin_bands = None
            break
        F = fftshift(fft2(data - np.mean(data)))
        perlin_bands.append(np.abs(F))
        
    if perlin_bands:
        rgb_fft_perlin = np.stack([
            perlin_bands[2] / np.max(perlin_bands[2]),
            perlin_bands[1] / np.max(perlin_bands[1]),
            perlin_bands[0] / np.max(perlin_bands[0])
        ], axis=-1)
        axes[0].imshow(rgb_fft_perlin)
        axes[0].set_title("(a) Perlin Noise Bands FFT")
        axes[0].axis('off')
    else:
        axes[0].text(0.5, 0.5, 'Perlin Data\nNot Found', ha='center', va='center', color='red')
        axes[0].axis('off')

    # --- Wavelet Noise ---
    wavelet_bands = []
    for o in octaves:
        data = load_raw_data(f"wavelet_noise_2D_octave_{o}.raw")
        if data is None: 
            print("Warning: Missing Wavelet data for Figure 9. Skipping Wavelet plot.")
            wavelet_bands = None
            break
        F = fftshift(fft2(data - np.mean(data)))
        wavelet_bands.append(np.abs(F))
        
    if wavelet_bands:
        rgb_fft_wavelet = np.stack([
            wavelet_bands[2] / np.max(wavelet_bands[2]),
            wavelet_bands[1] / np.max(wavelet_bands[1]),
            wavelet_bands[0] / np.max(wavelet_bands[0])
        ], axis=-1)
        axes[1].imshow(rgb_fft_wavelet)
        axes[1].set_title("(b) Wavelet Noise Bands FFT")
        axes[1].axis('off')
    else:
        axes[1].text(0.5, 0.5, 'Wavelet Data\nNot Found', ha='center', va='center', color='red')
        axes[1].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_filename = "result_analyze/figure9_comparison.png"
    plt.savefig(output_filename, dpi=150)
    print(f"   → Saved: {output_filename}")
    plt.show()

if __name__ == "__main__":
    print("=" * 80)
    print("ENHANCED UNIFIED WAVELET & PERLIN NOISE ANALYSIS")
    print("Paper Figure 8 & 9 Replication - Frequency Control Enabled")
    print("• Raw values analysis (no [0,1] normalization)")
    print("• Detailed 5-column individual analysis")
    print("• Octave comparison with proper frequency scaling")
    print("• Figure 8 & 9 comparison between Perlin and Wavelet noise")
    print("=" * 80)
    
    # Print statistics for wavelet octave data
    print("\n" + "="*80)
    print("WAVELET NOISE DATA STATISTICS SUMMARY")
    print("="*80)
    
    # Standard octave files
    octaves = [3, 4, 5]
    print("\n--- Wavelet Noise Octave Results with Frequency Scaling ---")
    for octave in octaves:
        print(f"\nOctave {octave} (frequency scale: {2**octave}x):")
        octave_files = [
            (f"wavelet_noise_2D_octave_{octave}.raw", f"  Wavelet 2D Octave {octave}"),
            (f"wavelet_noise_3Dsliced_octave_{octave}.raw", f"  Wavelet 3D Sliced Octave {octave}"),
            (f"wavelet_noise_3Dprojected_octave_{octave}.raw", f"  Wavelet 3D Projected Octave {octave}")
        ]
        
        for filename, description in octave_files:
            data = load_raw_data(filename)
            if data is not None:
                print(f"{description:35s}: mean={np.mean(data):8.4f}, std={np.std(data):8.4f}, min={np.min(data):8.4f}, max={np.max(data):8.4f}")
            else:
                print(f"{description:35s}: FILE NOT FOUND")
    
    # Print statistics for Perlin octave data
    print("\n--- Perlin Noise Octave Results with Frequency Scaling ---")
    for octave in octaves:
        print(f"\nOctave {octave} (frequency scale: {2**octave}x):")
        octave_files = [
            (f"perlin_noise_2D_octave_{octave}.raw", f"  Perlin 2D Octave {octave}"),
            (f"perlin_noise_3Dsliced_octave_{octave}.raw", f"  Perlin 3D Sliced Octave {octave}")
        ]
        
        for filename, description in octave_files:
            data = load_raw_data(filename)
            if data is not None:
                print(f"{description:35s}: mean={np.mean(data):8.4f}, std={np.std(data):8.4f}, min={np.min(data):8.4f}, max={np.max(data):8.4f}")
            else:
                print(f"{description:35s}: FILE NOT FOUND")
    
    # Generate Figure 8 and 9 comparisons first (main focus)
    print("\n" + "="*80)
    print("FIGURE 8 & 9 COMPARISON GENERATION")
    print("="*80)
    
    # Execute Figure 8 and 9 analyses
    create_figure8_comparison(octave=4)
    create_figure9_comparison(octaves=[3, 4, 5])
    
    # Generate detailed individual analysis for all files
    print("\nGenerating Detailed Individual Analysis (5-Column Layout)...")
    
    # All possible files to analyze (including Perlin noise now)
    all_files = []
    
    # Wavelet noise files
    for octave in [3, 4, 5]:
        octave_files = [
            (f"wavelet_noise_2D_octave_{octave}.raw", f"Wavelet 2D Octave {octave}"),
            (f"wavelet_noise_3Dsliced_octave_{octave}.raw", f"Wavelet 3D Sliced Octave {octave}"),
            (f"wavelet_noise_3Dprojected_octave_{octave}.raw", f"Wavelet 3D Projected Octave {octave}")
        ]
        all_files.extend(octave_files)
    
    # Perlin noise files
    for octave in [3, 4, 5]:
        octave_files = [
            (f"perlin_noise_2D_octave_{octave}.raw", f"Perlin 2D Octave {octave}"),
            (f"perlin_noise_3Dsliced_octave_{octave}.raw", f"Perlin 3D Sliced Octave {octave}")
        ]
        all_files.extend(octave_files)
    
    # Generate detailed analysis for each existing file
    for filename, description in all_files:
        # Check if file exists in result_raw directory
        file_path = f'result_raw/{filename}' if not filename.startswith('result_raw/') else filename
        if os.path.exists(file_path):
            print(f"   Analyzing: {description}")
            fig, peak_r = analyze_single_band(filename, description)
            if fig:
                output_name = f'result_analyze/{filename.replace(".raw", "")}_detailed_analysis.png'
                plt.savefig(output_name, dpi=150, bbox_inches='tight')
                print(f"   → Saved: {output_name}")
                plt.show()
        else:
            print(f"   Warning: Skipping {description}: file not found")
    
    # Check for standard octave files and generate comparisons
    octaves_to_analyze = [3, 4, 5]
    available_octaves = []
    
    for oct in octaves_to_analyze:
        files = [
            f"result_raw/wavelet_noise_2D_octave_{oct}.raw",
            f"result_raw/wavelet_noise_3Dsliced_octave_{oct}.raw",
            f"result_raw/wavelet_noise_3Dprojected_octave_{oct}.raw"
        ]
        if all(os.path.exists(f) for f in files):
            available_octaves.append(oct)
    
    if available_octaves:
        print(f"\nGenerating Wavelet Octave Comparisons for octaves: {available_octaves}")
        for oct in available_octaves:
            create_octave_comparison_figure(oct)
    else:
        print("\nWarning: No complete wavelet octave datasets found, skipping octave analysis.")
    
    print("\n" + "=" * 80)
    print("ENHANCED ANALYSIS COMPLETE")
    print("Generated outputs in result_analyze/:")
    print("• Figure 8 comparison (result_analyze/figure8_comparison_octave_4.png)")
    print("• Figure 9 comparison (result_analyze/figure9_comparison.png)")
    print("• Detailed individual analysis (result_analyze/*_detailed_analysis.png)")
    print("• Wavelet octave comparisons (result_analyze/wavelet_full_comparison_octave_*.png)")
    print(f"\nKey observations from paper replication:")
    print("• Figure 8: Shows band-limiting differences between Perlin and Wavelet noise")
    print("• Figure 9: Demonstrates frequency band separation in RGB colormap")
    print("• Perlin noise: Shows broad frequency content with some aliasing")
    print("• Wavelet noise: Clean band-limited frequency content")
    print("• 3D projections preserve band-limiting better than 3D slicing")
    print("• Frequency scaling properly controls noise detail level")
    print("=" * 80) 