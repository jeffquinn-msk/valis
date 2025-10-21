import os
import re

import geopandas
import pandas
import pyvips
import shapely
from ome_types import from_xml
from typing import Dict
from typing import Generator, Tuple
import numpy as np
from scipy.stats import pearsonr
import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.affinity import scale
from sklearn.preprocessing import StandardScaler

VIPS_FORMAT_NUMPY_DTYPE = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}

def iterate_tiles(
        image: pyvips.Image,
        tile_size: int = 256,
        overlap: int = 0
) -> Generator[Tuple[np.ndarray, int, int], None, None]:
    """
    Iterate over a pyvips.Image in tiles.

    Parameters:
    -----------
    image : pyvips.Image
        The image to tile
    tile_size : int
        Size of each tile in pixels (default: 256)
    overlap : int
        Number of pixels to overlap between adjacent tiles (default: 0)

    Yields:
    -------
    tuple : (tile_array, x, y, width, height)
        tile_array : np.ndarray
            The cropped tile as a numpy array
        x : int
            X coordinate of the tile's top-left corner
        y : int
            Y coordinate of the tile's top-left corner
        width : int
            Actual width of the tile (may be smaller at edges)
        height : int
            Actual height of the tile (may be smaller at edges)
    """
    step = tile_size - overlap

    for y in range(0, image.height, step):
        for x in range(0, image.width, step):
            # Calculate actual tile dimensions (handle edge cases)
            width = min(tile_size, image.width - x)
            height = min(tile_size, image.height - y)

            # Extract the tile and convert to numpy array
            tile = image.crop(x, y, width, height)
            tile_array = tile.numpy()

            yield tile_array, x, y


def read_ome_tiff(file_path: str) -> Dict[str, pyvips.Image]:
    """
    Read an OME-TIFF file and return a dictionary mapping channel names to PyVips Image objects.

    Parameters:
    -----------
    file_path : str
        Path to the OME-TIFF file

    Returns:
    --------
    Dict[str, pyvips.Image]
        Dictionary mapping channel names to pyvips.Image objects
    """
    # Load the first page to get the OME-XML metadata
    first_page = pyvips.Image.new_from_file(file_path, page=0)

    # Extract OME-XML metadata from the image description
    ome_xml = first_page.get("image-description")

    # Parse the OME-XML
    ome = from_xml(ome_xml)

    # Get the first image (assuming single image in the OME-TIFF)
    image = ome.images[0]
    pixels = image.pixels

    # Create dictionary to store channel name -> image mapping
    channel_dict = {}

    # Iterate through each channel
    for channel_idx, channel in enumerate(pixels.channels):
        # Get the channel name
        channel_name = channel.name if channel.name else f"Channel_{channel_idx}"

        # Load the corresponding page/IFD
        # The IFD index corresponds to the channel index in the OME-TIFF structure
        channel_image = pyvips.Image.new_from_file(file_path, page=channel_idx)

        # Add to dictionary
        channel_dict[channel_name] = channel_image

    return channel_dict

def compare_two_images_tilewise(images: dict, protein_name, sigma=10):
    cosmx_name = sorted([x for x in images.keys() if protein_name.lower() in x.lower()])[0]
    cosmx_image = images[cosmx_name]
    mif_image = images[f"mIF_{protein_name}"]
    if mif_image.height != cosmx_image.height or mif_image.width != cosmx_image.width:
        raise RuntimeError()

    mif_image_tiles = iterate_tiles(mif_image, tile_size=256, overlap=64)
    cosmx_image_tiles = iterate_tiles(cosmx_image, tile_size=256, overlap=64)

    results = []

    for (cosmx_tile, _, _), (mif_tile, x, y) in tqdm.tqdm(zip(cosmx_image_tiles, mif_image_tiles)):
        results.append({
            "x":x,
            "y":y,
            "cosmx_mean": mif_tile.mean(),
            "mIF_mean": cosmx_tile.mean(),
            "cosmx_max": mif_tile.max(),
            "mIF_max": cosmx_tile.max(),
            "protein":protein_name
        })
    return pandas.DataFrame(results)


from shapely.geometry import Polygon


def paint_polygon_on_image(image, polygon, color="#000000", opacity=0.8):
    """
    Paint a Shapely polygon onto a pyvips.Image using SVG

    Args:
        image: pyvips.Image to paint on
        polygon: shapely.geometry.Polygon to paint
        color: hex color string like "#ff0000"
        opacity: float between 0 and 1

    Returns:
        pyvips.Image with polygon painted
    """
    # Get image dimensions
    width = image.width
    height = image.height


    svg_content = polygon.svg()

    # Replace all fill attributes with transparent
    svg_content = re.sub(r'fill="[^"]*"', 'fill="none"', svg_content)
    svg_content = re.sub(r'fill-rule="[^"]*"', '', svg_content)

    # Replace all stroke attributes with our color
    svg_content = re.sub(r'stroke="[^"]*"', f'stroke="{color}"', svg_content)

    # Wrap it in a complete SVG document
    svg = f"""
    <svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">
      {svg_content}
    </svg>
    """

    # Load SVG as pyvips image
    svg_image = pyvips.Image.svgload_buffer(svg.encode())

    # Ensure proper dimensions
    if svg_image.width != width or svg_image.height != height:
        svg_image = svg_image.resize(width / svg_image.width)

    # Add alpha channel if needed
    if image.bands == 3:
        image = image.bandjoin(255)
    if svg_image.bands == 3:
        svg_image = svg_image.bandjoin(255)

    # Composite
    result = image.composite(svg_image, 'over')

    return result


def create_colorbar_with_labels(height, width=100, cmap_name='coolwarm', vmin=-1, vmax=1, label="Rank Difference"):
    """
    Create a colorbar with ticks and labels using matplotlib
    """
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import tempfile
    import os

    # Create figure with colorbar
    fig = Figure(figsize=(width / 100, height / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0.1, 0.05, 0.3, 0.9])  # [left, bottom, width, height]

    # Create colorbar
    cmap = plt.get_cmap(cmap_name)
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax)
    cb.set_label(label, rotation=270, labelpad=20)

    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, "bar.tiff")
    fig.savefig(tmp_path, dpi=100, bbox_inches='tight')

    plt.close(fig)

    # Read back with pyvips
    colorbar = pyvips.Image.new_from_file(tmp_path)
    # Resize to desired dimensions
    scale_x = width / colorbar.width
    scale_y = height / colorbar.height
    colorbar = colorbar.resize(scale_x, vscale=scale_y)

    return colorbar


def compare_two_images_tilewise_with_heatmap(
        cosmx_image,
        mif_image,
        protein_name,
        meaningful_overlaps: geopandas.GeoDataFrame,
        tile_size=256,
        overlap=64,
        plot_scale_factor=32,
        baseline_rank_differences:pandas.DataFrame=None):
    if mif_image.height != cosmx_image.height or mif_image.width != cosmx_image.width:
        raise RuntimeError("Images must have the same dimensions")

    width = cosmx_image.width
    height = cosmx_image.height

    # Create downsampled heatmap array
    heatmap_height = (height + plot_scale_factor - 1) // plot_scale_factor
    heatmap_width = (width + plot_scale_factor - 1) // plot_scale_factor
    heatmap_array = np.zeros((heatmap_height, heatmap_width), dtype=np.float32)
    count_array = np.zeros((heatmap_height, heatmap_width), dtype=np.int32)

    mif_image_tiles = iterate_tiles(mif_image, tile_size=tile_size, overlap=overlap)
    cosmx_image_tiles = iterate_tiles(cosmx_image, tile_size=tile_size, overlap=overlap)

    results = []

    for (cosmx_tile, cx, cy), (mif_tile, mx, my) in tqdm.tqdm(zip(cosmx_image_tiles, mif_image_tiles)):
        tile_shape = shapely.box(cx, cy, cx + tile_size, cy + tile_size)
        if not meaningful_overlaps.contains(tile_shape).any():
            continue
        cosmx_mean = cosmx_tile.mean()
        mif_mean = mif_tile.mean()
        diff = cosmx_mean - mif_mean
        # Map tile coordinates to heatmap coordinates
        tile_width = cosmx_tile.shape[1]
        tile_height = cosmx_tile.shape[0]

        results.append({
            "x": cx,
            "y": cy,
            "tile_width": tile_width,
            "tile_height": tile_height,
            "cosmx_mean": cosmx_mean,
            "mIF_mean": mif_mean,
            "cosmx_max": cosmx_tile.max(),
            "mIF_max": mif_tile.max(),
            "difference": diff,
            "protein": protein_name
        })
    results = pandas.DataFrame(results)

    if baseline_rank_differences is not None:
        baseline_rank_differences = baseline_rank_differences[["x", "y", "rank_diff"]].rename(
            columns={
                "rank_diff": "baseline_rank_diff"
            }
        )
        results = results.merge(
            baseline_rank_differences,
            left_on=["x", "y"],
            right_on=["x", "y"],
            how="left"
        ).dropna(subset=["baseline_rank_diff"])

    results['cosmx_mean_rank'] = results.cosmx_mean.rank()
    results['mIF_mean_rank'] = results['mIF_mean'].rank()
    results['rank_diff'] = results['cosmx_mean_rank'] - results['mIF_mean_rank']
    if baseline_rank_differences is not None:
        results['rank_diff'] = results['rank_diff'] - results['baseline_rank_diff']
    results['rank_diff_norm'] = StandardScaler().fit_transform((results['cosmx_mean_rank'] - results['mIF_mean_rank']).values[:,None]).flatten()

    for _, row in results.iterrows():
        cx = row['x']
        cy = row['y']
        tile_width = row['tile_width']
        tile_height = row['tile_height']
        rank_diff = row['rank_diff']
        heatmap_x_start = cx // plot_scale_factor
        heatmap_y_start = cy // plot_scale_factor
        heatmap_x_end = (cx + tile_width + plot_scale_factor - 1) // plot_scale_factor
        heatmap_y_end = (cy + tile_height + plot_scale_factor - 1) // plot_scale_factor

        # Add difference to heatmap (accumulate for averaging overlaps)
        heatmap_array[heatmap_y_start:heatmap_y_end, heatmap_x_start:heatmap_x_end] += rank_diff
        count_array[heatmap_y_start:heatmap_y_end, heatmap_x_start:heatmap_x_end] += 1

    # Average overlapping regions
    mask = count_array > 0
    heatmap_array[mask] /= count_array[mask]

    # Normalize to [-1, 1] range
    theoretical_max_abs_diff = len(results)
    normalized_heatmap = heatmap_array / theoretical_max_abs_diff

    # Apply coolwarm colormap
    cmap = plt.get_cmap('coolwarm')
    # Map from [-1, 1] to [0, 1]
    color_values = (normalized_heatmap + 1) / 2
    rgba_heatmap = cmap(color_values)

    # Convert to RGB uint8 (H, W, 3)
    rgb_heatmap = (rgba_heatmap[:, :, :3] * 255).astype(np.uint8)

    # Convert numpy array to pyvips image
    diff_image = pyvips.Image.new_from_memory(
        rgb_heatmap.tobytes(),
        heatmap_width,
        heatmap_height,
        bands=3,
        format='uchar'
    )

    for shape in meaningful_overlaps.geometry.tolist():
        # Scale down by the factor (origin is at (0,0) by default)
        scaled = scale(shape, xfact=heatmap_width/width, yfact=heatmap_height/height, origin=(0,0))

        diff_image = paint_polygon_on_image(
            diff_image, scaled
        )

    colorbar_img = create_colorbar_with_labels(
        width=0.1*heatmap_width,
        height=heatmap_height,
        vmin=-theoretical_max_abs_diff,
        vmax=theoretical_max_abs_diff,
        label=f"Rank Difference {protein_name} (CosMX - mIF)"
    )

    # Create a text image
    text_height = 100
    text_image = pyvips.Image.text(
        f"Rank Diff {protein_name} (CosMX - mIF)",
        font="Arial",  # Specify font and size
        width=heatmap_width,  # Set the width of the text area
        height=text_height,  # Set the height of the text area
        align="centre",  # Align the text (left, centre, right)
        dpi=300  # Set the resolution for rendering
    )
    alpha = text_image.gravity('centre', heatmap_width, text_height)
    text_image = alpha.new_from_image([0, 0, 0]).bandjoin(alpha)
    original_diff_image_width = diff_image.width
    text_background = pyvips.Image.black
    diff_image = diff_image.insert(text_image, 0, 0)

    diff_image = pyvips.Image.arrayjoin([diff_image, colorbar_img], across=2)
    diff_image = diff_image.crop(0, 0, (original_diff_image_width + colorbar_img.width), diff_image.height)
    return results, diff_image

def grayscale_to_rgb(grayscale_image,
                     color="white"):

    # Ensure the image is indeed grayscale (1 band)
    if grayscale_image.bands != 1:
        raise ValueError("Input image is not a grayscale image (expected 1 band).")

    # Replicate the single band three times to create an RGB image
    # .bandjoin() combines images band-wise.
    # We join the grayscale image with itself twice.
    rgb_image = grayscale_image.flatten().colourspace('srgb')

    return rgb_image


def plot_tile_with_minimap(
        img: pyvips.Image,
        x: int,
        y: int,
        width: int,
        height: int,
        minimap_size: int = 200,
):
    # Extract tiles
    tile = img.crop(x, y, width, height)

    result = grayscale_to_rgb(tile)

    scale = min(minimap_size / img.width, minimap_size / img.height)
    minimap = grayscale_to_rgb(img.resize(scale))

    # Draw red rectangle on minimap
    rect_x = int(x * scale)
    rect_y = int(y * scale)
    rect_width = max(int(width * scale), 2)
    rect_height = max(int(height * scale), 2)

    minimap = minimap.draw_rect([255,0,0], rect_x, rect_y, rect_width, rect_height)

    result = result.insert(minimap, 5, 5)
    result = result.draw_rect([255,255,255], 4, 4, minimap.width+1, minimap.height+1)

    # Save result
    return result


def plot_tile_comparison_with_minimap(
        img1: pyvips.Image,
        img2: pyvips.Image,
        x: int,
        y: int,
        width: int,
        height: int,
        output_file: str,
        minimap_size: int = 200,
):
    # Extract tiles
    tile1 = plot_tile_with_minimap(
        img1,
        x=x,
        y=y,
        width=width,
        height=height
    )
    tile2 = plot_tile_with_minimap(
        img2,
        x=x,
        y=y,
        width=width,
        height=height
    )
    # Join tiles side by side
    combined_tiles = pyvips.Image.arrayjoin([tile1, tile2], across=2)

    # Save result
    combined_tiles.write_to_file(output_file)

def main():
    data_dir = "/Users/quinnj2/Documents/epithelioid_sarcoma/data/551"
    dst_dir = f"/Users/quinnj2/Documents/epithelioid_sarcoma/runs/valis_2025-10-20"

    meaningful_overlaps_fn = os.path.join(dst_dir, "meaningful_overlaps.geojson")
    meaningful_overlaps = geopandas.read_file(meaningful_overlaps_fn)
    mif_proteins = [
        "CD8",
        "SMA",
        "Her2",
        "FoxP3",
        "CD163",
    ]

    images = read_ome_tiff(os.path.join(dst_dir, "aligned.ome.tif"))

    dfs = []

    baseline_df, heatmap = compare_two_images_tilewise_with_heatmap(
        cosmx_image=images["COSMX_DAPI"], mif_image=images["mIF_DAPI"],
        protein_name="DAPI",
        meaningful_overlaps=meaningful_overlaps
    )
    dfs.append(baseline_df)
    res = pearsonr(baseline_df.cosmx_mean, baseline_df.mIF_mean)
    print(f"Pearsons R for positive control (DAPI): {res.statistic} (p: {res.pvalue})")
    print(os.path.join(dst_dir, f"mif_vs_cosmx_DAPI_diff.png"))
    heatmap.write_to_file(os.path.join(dst_dir, f"mif_vs_cosmx_DAPI_diff.png"))

    for protein_name in mif_proteins:
        print(protein_name)

        cosmx_name = sorted([x for x in images.keys() if protein_name.lower() in x.lower()])[0]
        cosmx_image = images[cosmx_name]
        mif_image = images[f"mIF_{protein_name}"]

        df, heatmap = compare_two_images_tilewise_with_heatmap(
            cosmx_image=cosmx_image, mif_image=mif_image,
            protein_name=protein_name,
            meaningful_overlaps=meaningful_overlaps,
            baseline_rank_differences=baseline_df
        )
        dfs.append(df)
        res = pearsonr(df.cosmx_mean, df.mIF_mean)
        print(f"Pearsons R for {protein_name}: {res.statistic} (p: {res.pvalue})")
        print(os.path.join(dst_dir, f"mif_vs_cosmx_{protein_name}_diff.png"))
        heatmap.write_to_file(os.path.join(dst_dir, f"mif_vs_cosmx_{protein_name}_diff.png"))

    df = pandas.concat(dfs)
    df.to_parquet(
        os.path.join(dst_dir, "mif_vs_cosmx.parquet")
    )

if __name__ == "__main__":
    main()