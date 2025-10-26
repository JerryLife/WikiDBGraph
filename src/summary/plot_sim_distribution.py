import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

def plot_similarity_distribution_area(pt_file_path):
    tensor = torch.load(pt_file_path, map_location="cpu")
    print("Loaded tensor with shape", tensor.shape)

    similarities = tensor[:, 2].numpy()
    labels = tensor[:, 3].numpy()

    sim_label1 = similarities[labels == 1]
    sim_label0 = similarities[labels == 0]

    print(f"Total: label=0 ({len(sim_label0)}), label=1 ({len(sim_label1)})")

    bins = np.linspace(-1, 1, 51)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    hist0, _ = np.histogram(sim_label0, bins=bins)
    hist1, _ = np.histogram(sim_label1, bins=bins)

    gamma = 0.2
    hist0_scaled = hist0.astype(float) ** gamma
    hist1_scaled = hist1.astype(float) ** gamma

    x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 400)
    y0_smooth = make_interp_spline(bin_centers, hist0_scaled)(x_smooth)
    y1_smooth = make_interp_spline(bin_centers, hist1_scaled)(x_smooth)

    plt.figure(figsize=(10, 6))
    plt.fill_between(x_smooth, y0_smooth, alpha=0.5, label=f"label=0 (n={len(sim_label0)})", color="blue")
    plt.fill_between(x_smooth, y1_smooth, alpha=0.5, label=f"label=1 (n={len(sim_label1)})", color="orange")

    plt.xlabel("Similarity")
    plt.ylabel(f"Count$^{{{gamma}}}$ (50 bins, smoothed)")
    plt.title("Similarity Distribution (Smoothed Area Plot with Power Transform)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

if __name__ == "__main__":
    pt_path = "/home/ziyangw/wkdbs_graph/out/graph/all_exhaustive_predictions.pt"
    plot_similarity_distribution_area(pt_path)
