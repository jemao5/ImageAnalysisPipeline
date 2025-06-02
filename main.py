import os
import numpy as np
import tifffile
from cellpose import models
import matplotlib.pyplot as plt
from skimage import io, color
import seaborn as sns
import nd2
import time

def nd2_to_tiff(nd2_path, output_root, strain_names):
    with nd2.ND2File(nd2_path) as f:
        axes = f.sizes  # {'T': ..., 'P': ..., 'Z': ..., 'Y': ..., 'X': ...}
        pixel_size_xy = f.metadata.channels[0].volume.axesCalibration[0] * f.metadata.channels[0].volume.axesCalibration[1]
        pixel_size_str = f"{pixel_size_xy:.6f}"
        n_timepoints = axes['T']
        n_positions = axes['P']
        n_z = axes['Z']
        n_strains = len(strain_names)
        assert n_positions % n_strains == 0, "Mismatch: positions must divide evenly into strains"
        positions_per_strain = n_positions // n_strains
        data = f.asarray()  # shape: (T, P, Z, Y, X)

        tiff_paths = []
        for strain_idx, strain in enumerate(strain_names):
            strain_dir = os.path.join(output_root, strain)
            os.makedirs(strain_dir, exist_ok=True)
            for local_pos in range(positions_per_strain):
                pos_idx = strain_idx * positions_per_strain + local_pos
                stack = data[:, pos_idx, :, :, :]  # shape: (T, Z, Y, X)
                out_path = os.path.join(strain_dir, f"field_{local_pos}.tiff")
                tifffile.imwrite(
                    out_path,
                    stack.astype(np.uint16),
                    imagej=True,
                    metadata={
                        "axes": "TZYX",
                        "pixel_size": pixel_size_str
                    }
                )
                tiff_paths.append((strain, out_path, float(pixel_size_str)))
    print("✅ ND2 extraction complete.")
    return tiff_paths


# def segment_and_measure(tiff_paths):
#     model = models.CellposeModel(pretrained_model='cyto3', gpu=False)
#     results = {}

#     for strain, path, pixel_area in tiff_paths:
#         images = tifffile.imread(path)  # shape: (T, Z, Y, X)
#         first_frame_zstack = images[0]  # shape: (Z, Y, X)
#         zproj = np.max(first_frame_zstack, axis=0)  # shape: (Y, X)

#         # plt.imshow(zproj, cmap='gray')
#         # plt.show()

#         start = time.time()
#         masks, _, _ = model.eval(
#             zproj,
#             diameter=30,                 # avoid auto-scale
#             flow_threshold=0.4,
#             cellprob_threshold=0.0,
#             normalize=True,
#             resample=True
#         )
#         print("Eval time:", time.time() - start)


#         # Save just this one mask
#         mask_path = path.replace(".tiff", "_mask.npy")
#         np.save(mask_path, masks)

#         # Compute cell sizes
#         sizes = [
#             (masks == c).sum() * pixel_area
#             for c in np.unique(masks)[1:]  # exclude background
#             if (masks == c).sum() * pixel_area < 160
#         ]

#         results.setdefault(strain, []).extend(sizes)

#     return results
def segment_and_measure(tiff_paths):
    MAX_AREA = 160
    model = models.CellposeModel(pretrained_model='cyto3', gpu=False)
    results = {}

    # Group TIFFs by strain
    from collections import defaultdict
    strain_to_paths = defaultdict(list)
    for strain, path, pixel_area in tiff_paths:
        strain_to_paths[strain].append((path, pixel_area))

    for strain, items in strain_to_paths.items():
        batch_images = []
        metadata = []

        for path, pixel_area in items:
            images = tifffile.imread(path)  # shape: (T, Z, Y, X)
            zproj = np.max(images[0], axis=0)  # (Y, X) from first frame
            zproj = (zproj - zproj.min()) / (zproj.max() - zproj.min() + 1e-8)
            zproj = (zproj * 255).astype(np.uint8)

            batch_images.append(zproj)
            metadata.append((path, pixel_area))

        print(f"▶ Segmenting {len(batch_images)} fields from {strain}...")
        masks_list, _, _ = model.eval(batch_images, diameter=None, normalize=True)

        strain_sizes = []
        for (path, pixel_area), masks in zip(metadata, masks_list):
            np.save(path.replace(".tiff", "_mask.npy"), masks)
            sizes = [
                (masks == c).sum() * pixel_area
                for c in np.unique(masks)[1:]
                if (masks == c).sum() * pixel_area < MAX_AREA
            ]
            strain_sizes.extend(sizes)

        results[strain] = strain_sizes

    return results



def plot_results(results, output_path):
    data = [results[k] for k in results]
    labels = list(results.keys())

    sns.set_theme(rc={'figure.figsize': (15, 8.27)})
    ax = sns.swarmplot(data=data, orient='v', zorder=10, size=2.3)
    ax.set(xlabel='Strain', ylabel='Cell Size (μm²)', title='Cell Sizes')
    ax.set_xticks(range(len(labels)), labels=labels)

    sns.boxplot(
        data=data,
        showmeans=True, meanline=True,
        meanprops={'color': 'k', 'ls': '-', 'lw': 1},
        medianprops={'visible': False},
        whiskerprops={'visible': False},
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=ax
    )

    plt.savefig(os.path.join(output_path, "graph.pdf"), dpi=1000)
    plt.close()




def visualize_mask_overlay(tiff_path, mask_path=None, output_path=None, alpha=0.15):
    """
    Overlay Cellpose masks on a max-projected image.

    Parameters:
    - tiff_path (str): Path to the TIFF file.
    - mask_path (str): Path to the corresponding .npy mask file. If None, uses tiff_path with "_mask.npy" suffix.
    - output_path (str): Where to save the output image. If None, saves as tiff_path with "_overlay.png" suffix.
    - alpha (float): Transparency level for mask overlay.
    """
    if mask_path is None:
        mask_path = tiff_path.replace(".tiff", "_mask.npy")
    if output_path is None:
        output_path = tiff_path.replace(".tiff", "_overlay.png")

    # Load TIFF and mask
    image = io.imread(tiff_path)  # (T, Z, Y, X)
    mask = np.load(mask_path)

    # Project the first timepoint and normalize
    im_max = np.max(image[0], axis=0)  # (Y, X)
    rescaled = (im_max - im_max.min()) / (im_max.max() - im_max.min() + 1e-8)

    # Generate random colors for each mask label
    num_labels = np.max(mask)
    rng = np.random.default_rng(42)  # Fixed seed for consistency
    colors = rng.uniform(0, 1, size=(num_labels, 3))

    # Overlay mask
    overlay = color.label2rgb(label=mask, image=rescaled, colors=colors, alpha=alpha, bg_label=0, bg_color=None)

    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"✅ Overlay saved to {output_path}")



# === Usage === #
if __name__ == "__main__":
    nd2_file = r"C:\Users\jonat\Documents\NYU_Reserach\ImageAnalysisPipeline\data\CN1_CN23_CN81_titan_recovery.nd2"  # <-- Set this to your ND2 file
    strain_names = ['strainA', 'strainB', 'strainC']  # One per timepoint
    output_root = r"C:\Users\jonat\Documents\NYU_Reserach\ImageAnalysisPipeline\data"

    tiff_paths = nd2_to_tiff(nd2_file, output_root, strain_names)
    results = segment_and_measure(tiff_paths)
    plot_results(results, output_root)
    # visualize_mask_overlay(r"C:\Users\jonat\Documents\NYU_Reserach\ImageAnalysisPipeline\data\strainA\field_0.tiff")
