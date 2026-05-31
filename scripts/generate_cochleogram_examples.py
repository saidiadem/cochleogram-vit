
import numpy as np
import matplotlib.pyplot as plt
import os

# Create a directory to save the images
output_dir = '/mnt/data/home/adem/Desktop/pfa/results/cochleogram_examples'
os.makedirs(output_dir, exist_ok=True)

# Define the file paths and labels
cochleograms_to_plot = {
    'Normal': '/mnt/data/home/adem/Desktop/pfa/data/processed/cochleograms/102_1b1_Ar_sc_Meditron_00025.npy',
    'Crackle': '/mnt/data/home/adem/Desktop/pfa/data/processed/cochleograms/106_2b1_Pl_mc_LittC2SE_00106.npy',
    'Wheeze': '/mnt/data/home/adem/Desktop/pfa/data/processed/cochleograms/106_2b1_Pr_mc_LittC2SE_00115.npy',
    'Both': '/mnt/data/home/adem/Desktop/pfa/data/processed/cochleograms/107_2b3_Ar_mc_AKGC417L_00132.npy'
}

# Generate and save the plots
for label, file_path in cochleograms_to_plot.items():
    # Load the cochleogram data
    cochleogram = np.load(file_path)

    # Create the plot
    fig, ax = plt.subplots(figsize=(6, 4))
    img = ax.imshow(cochleogram, aspect='auto', origin='lower', cmap='viridis')
    ax.set_title(f'Cochleogram Example: {label}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    # Save the figure
    output_path = os.path.join(output_dir, f'{label.lower()}_cochleogram.png')
    plt.savefig(output_path)
    plt.close(fig)
    print(f'Saved: {output_path}')
