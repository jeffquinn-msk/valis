import json
import pickle
import time
import numpy as np
import pandas
import shapely
from pandas.compat import pyarrow
from shapely import Polygon
from tqdm import tqdm

from valis import registration, slide_tools
import logging
import os
from typing import List, Optional
from valis import warp_tools
import pyvips

from ome_types import OME
from ome_types.model import Image, Pixels, Channel, TiffData, Plane
from ome_types.model.simple_types import PixelsID, ChannelID, ImageID, UnitsLength

logger = logging.getLogger(__name__)
def create_ome_metadata(
        width: int,
        height: int,
        protein_names: List[str],
        protein_ids: List[str],
        um_per_px: float,
        dtype: str = "uint16",
) -> str:
    """
    Create OME-XML metadata for single or multi-channel protein images.

    Parameters:
    -----------
    width : int
        Image width in pixels
    height : int
        Image height in pixels
    protein_names : list of str
        Human-readable protein name(s), one per channel
    protein_ids : list of str
        Protein identifier(s), one per channel
    um_per_px : float
        Micrometers per pixel
    dtype : str
        Data type (default: 'uint16')

    Returns:
    --------
    str : OME-XML metadata as string
    """

    # Validate inputs
    num_channels = len(protein_names)
    if len(protein_ids) != num_channels:
        raise ValueError(
            f"Length of protein_names ({num_channels}) and protein_ids "
            f"({len(protein_ids)}) must match"
        )

    if num_channels == 0:
        raise ValueError("Must provide at least one protein name and ID")

    # Create the OME structure
    ome = OME()

    # Create Channels
    channels = []
    for c in range(num_channels):
        channel = Channel(
            id=ChannelID(f"Channel:0:{c}"),
            name=protein_names[c],
            samples_per_pixel=1,
        )
        channels.append(channel)

    # Create TiffData blocks
    tiff_data_blocks = []
    for c in range(num_channels):
        tiff_data = TiffData(
            first_z=0,
            first_c=c,
            first_t=0,
            ifd=c,
            plane_count=1,
        )
        tiff_data_blocks.append(tiff_data)

    # Create Planes
    planes = []
    for c in range(num_channels):
        plane = Plane(
            the_z=0,
            the_c=c,
            the_t=0,
        )
        planes.append(plane)

    # Create Pixels
    pixels = Pixels(
        id=PixelsID("Pixels:0"),
        dimension_order="XYZCT",
        size_x=width,
        size_y=height,
        size_z=1,
        size_c=num_channels,
        size_t=1,
        type=dtype,
        big_endian=False,
        physical_size_x=um_per_px,
        physical_size_y=um_per_px,
        physical_size_x_unit=UnitsLength.MICROMETER,
        physical_size_y_unit=UnitsLength.MICROMETER,
        channels=channels,
        tiff_data_blocks=tiff_data_blocks,
        planes=planes,
    )

    # Create Image name from protein info
    if num_channels == 1:
        image_name = f"{protein_names[0]} ({protein_ids[0]})"
    else:
        # For multichannel, create a combined name
        image_name = f"Multichannel ({', '.join(protein_ids)})"

    # Create Image with pixels
    image = Image(
        id=ImageID("Image:0"),
        name=image_name,
        pixels=pixels,
    )

    # Assemble the structure
    ome.images.append(image)

    # Convert to XML string
    return ome.to_xml()

NUMPY_FORMAT_BF_DTYPE = {'uint8': 'uint8',
                         'int8': 'int8',
                         'uint16': 'uint16',
                         'int16': 'int16',
                         'uint32': 'uint32',
                         'int32': 'int32',
                         'float32': 'float',
                         'float64': 'double'}

def vips2bf_dtype(vips_format):
    """Get bioformats equivalent of the pyvips pixel type

    Parameters
    ----------
    vips_format : str
        Format of the pyvips.Image

    Returns
    -------
    bf_dtype : str
        String format of Bioformats datatype

    """

    np_dtype = slide_tools.VIPS_FORMAT_NUMPY_DTYPE[vips_format]
    bf_dtype = NUMPY_FORMAT_BF_DTYPE[str(np_dtype().dtype)]

    return bf_dtype


def create_progress_callback():
    """Create a progress callback for writing the output file."""
    pbar = tqdm(total=100, desc="Writing OME-TIFF", unit="%")
    last_update = time.time()

    def eval_callback(image, progress):
        if (time.time() - last_update) > 0.25:
            # Set the progress bar's current iteration count directly
            pbar.n = progress.percent

            # Refresh the display to show the updated value
            pbar.refresh()

    eval_callback.close = lambda: pbar.close()
    return eval_callback


def write_ome_tiff(
    images: list[pyvips.Image],
    names: list[str],
    output_path,
    um_per_px: float,
    scale: float = 1.0,
):
    if scale < 1.0:
        images = [im.resize(scale) for im in images]

    # Only create OME metadata if ome_tiff is True
    ome_xml = create_ome_metadata(
            width=images[0].width,
            height=images[0].height,
            protein_names=names,
            protein_ids=names,
            um_per_px=um_per_px / scale,
            dtype=vips2bf_dtype(images[0].format),
        )

    stacked = pyvips.Image.arrayjoin(images, across=1)
    stacked.set_type(pyvips.GValue.gstr_type, "image-description", ome_xml)
    stacked.set_type(pyvips.GValue.gint_type, "page-height", images[0].height)

    stacked.set_progress(True)
    stacked.signal_connect("eval", create_progress_callback())

    # Append remaining images
    stacked.tiffsave(
            output_path,
            pyramid=True,
            subifd=True,
            tile=True,
            compression="jpeg",
            tile_height=256,
            tile_width=256,
            Q=100,
            bigtiff=True,
        )

    print(f"\nSuccessfully wrote: {output_path}")


def setup_valis_logging():
    """Configure detailed stream logging for valis logger"""
    # Get the valis logger
    valis_logger = logging.getLogger('valis')
    valis_logger.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create detailed formatter with line number, function name, etc.
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s() - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)

    # Add handler to logger
    valis_logger.addHandler(console_handler)
    logger.addHandler(console_handler)

    return valis_logger

def preprocess_full_tma_image(image, bottom_percent=10):
    """
    Process an image by setting the bottom percentage to black and reflecting horizontally.

    Args:
        image: pyvips.Image object
        bottom_percent: Percentage of bottom to set to black (default: 10)

    Returns:
        pyvips.Image: Processed image
    """
    # Calculate the height of the bottom region
    height = image.height
    bottom_height = int(height * (bottom_percent / 100))

    # Calculate where the black region should start
    start_y = height - bottom_height

    # Create a black region
    black_region = pyvips.Image.black(image.width, bottom_height)

    # Extract the top portion of the original image
    top_portion = image.crop(0, 0, image.width, start_y)

    # Join the top portion with the black region
    result = top_portion.join(black_region, 'vertical')

    return result

def preprocess_cosmx_image(image, bottom_percent=10):
    # Reflect horizontally (flip left-right)
    result = image.fliphor()

    return result


def warp_new_slide(
        new_img: pyvips.Image,
        registrar: registration.Valis,
        source_img: str,
        dst_img: str):
    src_slide_obj = registrar.get_slide(source_img)
    dst_slide_obj = registrar.get_slide(dst_img)
    warped_img = warp_tools.warp_img(
        img=new_img,
        M=src_slide_obj.M,  # 3x3 affine transformation matrix
        bk_dxdy=src_slide_obj.bk_dxdy,  # Non-rigid displacement field (or None for rigid only)
        out_shape_rc=dst_slide_obj.slide_shape_rc,  # Output shape
        transformation_src_shape_rc=src_slide_obj.processed_img_shape_rc,  # Source shape
        transformation_dst_shape_rc=src_slide_obj.reg_img_shape_rc,  # Destination shape
        bbox_xywh=None,  # Optional crop bounding box
        bg_color=src_slide_obj.bg_color,  # Background color for empty areas
        interp_method="bicubic"
    )

    return warped_img




def main():
    # Setup logging first
    setup_valis_logging()
    data_dir = "/Users/quinnj2/Documents/epithelioid_sarcoma/data/551"

    mif_channels = [
        "DAPI",
        "CD8",
        "SMA",
        "Her2",
        "FoxP3",
        "CD163",
    ]

    cosmx_files = [
        "CD163.ome.tif",
        "CD8.ome.tif",
        "Her2.ome.tif",
        "FOXP3.ome.tif",
        "SMA.ome.tif",
    ]

    ref_img = os.path.join(data_dir, "morphology_focus/DAPI.tif")
    cosmx_source_image_fn = os.path.join(data_dir, "COSMX_DNA.tif")

    mif_img = os.path.join(data_dir, "R1-S2-2025-09-09T12-55-47.ome.tiff")

    adjusted_ref_image_fn = os.path.join(data_dir, "XENIUM_DAPI_transformed.tiff")
    adjusted_mif_image_fn = os.path.join(data_dir, "mIF_DAPI_transformed.tiff")
    adjusted_cosmx_image_fn = os.path.join(data_dir, "COSMX_transformed.tiff")
    if not os.path.exists(adjusted_ref_image_fn):
        img = pyvips.Image.new_from_file(ref_img, page=0)
        preprocess_full_tma_image(img, 18).write_to_file(adjusted_ref_image_fn)

    if not os.path.exists(adjusted_mif_image_fn):
        img = pyvips.Image.new_from_file(mif_img, page=0)
        preprocess_full_tma_image(img, 25).write_to_file(adjusted_mif_image_fn)

    if not os.path.exists(adjusted_cosmx_image_fn):
        img = pyvips.Image.new_from_file(cosmx_source_image_fn, page=0)
        preprocess_cosmx_image(img).write_to_file(adjusted_cosmx_image_fn)


    dst_dir = f"/Users/quinnj2/Documents/epithelioid_sarcoma/runs/valis_2025-10-13"

    if os.path.exists(os.path.join(dst_dir, "551/data/551_registrar.pickle")):
        logger.info(f"Using existing solution at {dst_dir}")
        with open(os.path.join(dst_dir, "551/data/551_registrar.pickle"), "rb") as f:
            registrar = pickle.load(f)
    else:
        # Create a Valis object and use it to register the slides in slide_src_dir
        registrar = registration.Valis(src_dir="/Users/quinnj2/Documents/epithelioid_sarcoma/data/551",
                                       dst_dir=dst_dir,
                                       img_list=[adjusted_cosmx_image_fn, adjusted_mif_image_fn, adjusted_ref_image_fn],
                                       reference_img_f=adjusted_ref_image_fn,
                                       thumbnail_size=4096,
                                       check_for_reflections=False,
                                       similarity_metric="euclidean",
                                       align_to_reference=True)
        registrar.register()

    if not os.path.exists(os.path.join(dst_dir, "aligned.ome.tif")):
        warped_slides = []
        names = []
        logger.info(f"Warping xenium dapi")
        img = pyvips.Image.new_from_file(adjusted_ref_image_fn, page=0)
        preprocessed_image = preprocess_full_tma_image(img, 18)
        warped_img = warp_new_slide(
            preprocessed_image,
            registrar,
            source_img=adjusted_ref_image_fn,
            dst_img=adjusted_ref_image_fn
        )
        warped_slides.append(
            warped_img
        )
        names.append("XENIUM_DAPI")

        logger.info(f"Warping cosmx dapi")
        img = pyvips.Image.new_from_file(adjusted_cosmx_image_fn, page=0)
        warped_img = warp_new_slide(
            img,
            registrar,
            source_img=adjusted_cosmx_image_fn,
            dst_img=adjusted_ref_image_fn
        )
        warped_slides.append(
            warped_img
        )
        names.append("COSMX_DAPI")

        for c_idx, channel in enumerate(mif_channels):
            logger.info(f"Warping mIF {channel}")
            img = pyvips.Image.new_from_file(mif_img, page=c_idx)
            preprocessed_image = preprocess_full_tma_image(img, 25)
            warped_img = warp_new_slide(
                preprocessed_image,
                registrar,
                source_img=adjusted_mif_image_fn,
                dst_img=adjusted_ref_image_fn
            )
            warped_slides.append(
                warped_img
            )
            names.append("mIF_" + channel)

        for fn in cosmx_files:
            logger.info(f"Warping cosmx {fn}")
            name = fn.split(".")[0]
            fn = os.path.join(data_dir, fn)
            img = preprocess_cosmx_image(pyvips.Image.new_from_file(fn, page=0))
            img_warped = warp_new_slide(
                img,
                registrar,
                adjusted_cosmx_image_fn,
                adjusted_ref_image_fn,
            )
            warped_slides.append(img_warped)
            names.append("COSMX_" + name)
        logger.info(f"Writing to {os.path.join(dst_dir, "aligned.ome.tif")}")
        write_ome_tiff(
            images=warped_slides,
            names=names,
            output_path=os.path.join(dst_dir, "aligned.ome.tif"),
            um_per_px=0.2,
            scale=1,
        )

    if not os.path.exists(os.path.join(dst_dir, "TMA2_ES2_polygons_warped.csv")):
        cosmx_slide_obj = registrar.get_slide(adjusted_cosmx_image_fn)
        ref_slide_obj = registrar.get_slide(adjusted_ref_image_fn)

        boundaries = pandas.read_csv(os.path.join(data_dir, "TMA2_ES2_polygons.csv.gz"))
        boundaries = boundaries.dropna()
        warped_boundaries = cosmx_slide_obj.warp_xy_from_to(
            boundaries[["x_global_px","y_global_px"]],
            to_slide_obj=ref_slide_obj
        )
        boundaries['x_warped'] = warped_boundaries[:,0]
        boundaries['y_warped'] = warped_boundaries[:, 1]
        boundaries.to_parquet(os.path.join(dst_dir, "TMA2_ES2_polygons_warped.csv"))

if __name__ == '__main__':
    main()
