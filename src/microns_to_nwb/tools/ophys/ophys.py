import numpy as np
import logging
from hdmf.backends.hdf5 import H5DataIO
from microns_phase3 import nda, utils
from pynwb.base import Images
from pynwb.image import GrayscaleImage
from pynwb.ophys import (
    RoiResponseSeries,
    Fluorescence,
    ImageSegmentation,
    OpticalChannel,
)

from microns_to_nwb.tools.cave_client import get_functional_coreg_table
from microns_to_nwb.tools.nwb_helpers import check_module

logger = logging.getLogger(__name__)

def add_summary_images(field_key, nwb):
    ophys = check_module(nwb, "ophys")

    correlation_image_data, average_image_data = (nda.SummaryImages & field_key).fetch1("correlation", "average")

    # The image dimensions are (height, width), for NWB it should be transposed to (width, height).
    correlation_image_data = correlation_image_data.transpose(1, 0)
    correlation_image = GrayscaleImage(
        name="correlation",
        data=correlation_image_data,
    )
    average_image_data = average_image_data.transpose(1, 0)
    average_image = GrayscaleImage(
        name="average",
        data=average_image_data,
    )

    # Add images to Images container
    segmentation_images = Images(
        name=f"SegmentationImages{field_key['field']}",
        images=[correlation_image, average_image],
        description=f"Correlation and average images for field {field_key['field']}.",
    )
    ophys.add(segmentation_images)


def add_plane_segmentation(field_key, nwb, imaging_plane, image_segmentation):
    image_height, image_width = (nda.Field & field_key).fetch1("px_height", "px_width")
    mask_pixels, mask_weights, unit_ids, mask_types = (nda.Segmentation * nda.MaskClassification * nda.ScanUnit.proj('field', 'mask_id') & field_key).fetch(
        "pixels", "weights", "unit_id", "mask_type", order_by="unit_id"
    )

    plane_segmentation = image_segmentation.create_plane_segmentation(
        name=f"PlaneSegmentation{field_key['field']}",
        description=f"The output from segmenting field {field_key['field']} contains "
        "the image masks (weights and mask classification) and the structural "
        f"ids extracted from the CAVE database on {nwb.file_create_date[0].strftime('%Y-%m-%d')}. "
        "To access the latest revision from the live resource see "
        "the notebook that is linked to the dandiset. The structual ids "
        "might not exist for all plane segmentations.",
        imaging_plane=imaging_plane,
        id=unit_ids,
    )

    # Reshape masks
    masks = utils.reshape_masks(mask_pixels, mask_weights, image_height, image_width)
    # The masks dimensions are (height, width, number of frames), for NWB it should be
    # transposed to (number of frames, width, height)
    masks = masks.transpose(2, 1, 0)

    # Add image masks
    plane_segmentation.add_column(
        name="image_mask",
        description="The image masks for each ROI.",
        data=H5DataIO(masks, compression=True),
    )

    # Add type of ROIs
    plane_segmentation.add_column(
        name="mask_type",
        description="The classification of mask as soma or artifact.",
        data=mask_types.astype(str),
    )

    return plane_segmentation


def add_functional_coregistration_to_plane_segmentation(
    field_key,
    plane_segmentation,
):
    # Get functional coregistration table from CAVE for this field
    materialization_ver, functional_coreg_table = get_functional_coreg_table(field_key=field_key)
    
    if functional_coreg_table.empty:
        return

    if not functional_coreg_table.unit_id.is_unique:
        logger.warning(
            f"WARNING: Functional coregistration table for field {field_key['field']} contains duplicate unit_ids."
            "Only the first occurrence will be used."
        )
        functional_coreg_table = functional_coreg_table.drop_duplicates(subset=["unit_id"], keep="first")

    # filter down to units for this field
    field_df = (nda.ScanUnit & field_key).fetch(format='frame').reset_index()

    # merge to coregistration table
    merge_df = field_df.merge(functional_coreg_table, how='left')
    
    # subset to desired columns
    subset_df = merge_df[['id', 'pt_supervoxel_id', 'pt_root_id', 'target_id', 'pt_position_x', 'pt_position_y', 'pt_position_z']].astype(np.float64)

    plane_segmentation.add_column(
        name="cave_ids",
        description=f"The identifier(s) in CAVE for field {field_key['field']}.",
        data=np.expand_dims(subset_df.id.tolist(), 1).tolist(), # cave_ids should be shape (n x 1)
        index=True,
    )

    plane_segmentation.add_column(
        name="pt_supervoxel_id",
        description=f"The ID of the supervoxel from the watershed segmentation that is under the pt_position (v{materialization_ver}).",
        data=subset_df.pt_supervoxel_id.tolist(),
    )

    plane_segmentation.add_column(
        name="pt_root_id",
        description=f"The ID of the segment/root_id under the pt_position from the Proofread Segmentation (v{materialization_ver}).",
        data=subset_df.pt_root_id.tolist(),
    )

    plane_segmentation.add_column(
        name="nucleus_id",
        description=f"The ID of the nucleus_id from CAVE table `nucleus_detection_v0`(v{materialization_ver}).",
        data=subset_df.target_id.tolist(),
    )

    plane_segmentation.add_column(
        name="pt_x_position",
        description=f"The x location in 4,4,40 nm voxels at a cell body for the cell (v{materialization_ver}).",
        data=subset_df.pt_position_x.tolist(),
    )

    plane_segmentation.add_column(
        name="pt_y_position",
        description=f"The y location in 4,4,40 nm voxels at a cell body for the cell (v{materialization_ver}).",
        data=subset_df.pt_position_y.tolist(),
    )

    plane_segmentation.add_column(
        name="pt_z_position",
        description=f"The z location in 4,4,40 nm voxels at a cell body for the cell (v{materialization_ver}).",
        data=subset_df.pt_position_z.tolist(),
    )


def _get_fluorescence(nwb, fluorescence_name):
    ophys = check_module(nwb, "ophys")

    if fluorescence_name in ophys.data_interfaces:
        return ophys.get(fluorescence_name)

    fluorescence = Fluorescence(name=fluorescence_name)
    ophys.add(fluorescence)

    return fluorescence

def add_roi_response_series(field_key, nwb, plane_segmentation, timestamps):
    # add Fluorescence traces
    traces_for_each_mask = (nda.Fluorescence * nda.ScanUnit.proj('field', 'mask_id') & field_key).fetch("trace", order_by="unit_id")
    continuous_traces = np.vstack(traces_for_each_mask).T

    roi_table_region = plane_segmentation.create_roi_table_region(
        region=list(range(continuous_traces.shape[1])), description=f"all rois in field {field_key['field']}"
    )

    roi_response_series = RoiResponseSeries(
        name=f"RoiResponseSeries{field_key['field']}",
        description=f"The fluorescence traces for field {field_key['field']}",
        data=H5DataIO(continuous_traces, compression=True),
        rois=roi_table_region,
        timestamps=H5DataIO(timestamps, compression=True),
        unit="n.a.",
    )

    fluorescence = _get_fluorescence(nwb=nwb, fluorescence_name="Fluorescence")
    fluorescence.add_roi_response_series(roi_response_series)

def add_deconvolved_roi_series(
    field_key,
    nwb,
    plane_segmentation,
    timestamps,
):
    """
    Store deconvolved per-ROI activity as a RoiResponseSeries linked to the same ROIs
    as Fluorescence. This mirrors your Fluorescence storage pattern.
    """
    traces_for_each_mask = (nda.Activity * nda.ScanUnit.proj('field', 'mask_id') & field_key).fetch("trace", order_by="unit_id")
    continuous_traces = np.vstack(traces_for_each_mask).T

    roi_table_region = plane_segmentation.create_roi_table_region(
        region=list(range(continuous_traces.shape[1])), description=f"all rois in field {field_key['field']}"
    )

    roi_response_series = RoiResponseSeries(
        name=f"DeconvolvedActivity{field_key['field']}",
        description="Per-ROI deconvolved activity aligned to fluorescence traces.",
        data=H5DataIO(continuous_traces, compression=True),
        rois=roi_table_region,
        timestamps=H5DataIO(timestamps, compression=True),
        unit="n.a.",
    )

    # Reuse the Fluorescence container
    fluorescence = _get_fluorescence(nwb=nwb, fluorescence_name="Fluorescence")
    fluorescence.add_roi_response_series(roi_response_series)

def add_ophys(scan_key, nwb, timestamps):
    device = nwb.create_device(
        name="Microscope",
        description="two-photon random access mesoscope",
    )
    ophys = nwb.create_processing_module("ophys", "processed 2p data")
    image_segmentation = ImageSegmentation()
    ophys.add(image_segmentation)

    all_field_data = (nda.Field & scan_key).fetch(as_dict=True)
    for field_data in all_field_data:
        optical_channel = OpticalChannel(
            name="OpticalChannel",
            description="an optical channel",
            emission_lambda=500.0,
        )
        field_x_in_meters = field_data["field_x"] / 1e6
        field_y_in_meters = field_data["field_y"] / 1e6
        field_z_in_meters = field_data["field_z"] / 1e6
        field_width_in_meters = field_data["um_width"] / 1e6
        field_height_in_meters = field_data["um_height"] / 1e6
        imaging_plane = nwb.create_imaging_plane(
            name=f"ImagingPlane{field_data['field']}",
            optical_channel=optical_channel,
            imaging_rate=np.nan,
            description=f"The imaging plane for field {field_data['field']} at {field_z_in_meters} meters depth.",
            device=device,
            excitation_lambda=920.0,
            indicator="GCaMP6",
            location="VISp,VISrl,VISlm,VISal",
            grid_spacing=[
                field_width_in_meters / field_data["px_width"],
                field_height_in_meters / field_data["px_height"],
            ],
            grid_spacing_unit="meters",
            origin_coords=[field_x_in_meters, field_y_in_meters, field_z_in_meters],
            origin_coords_unit="meters",
        )

        field_key = {**scan_key, **dict(field=field_data["field"])}

        plane_segmentation = add_plane_segmentation(field_key, nwb, imaging_plane, image_segmentation)
        add_functional_coregistration_to_plane_segmentation(
            field_key=field_key,
            plane_segmentation=plane_segmentation,
        )
        add_roi_response_series(field_key, nwb, plane_segmentation, timestamps)
        add_deconvolved_roi_series(field_key, nwb, plane_segmentation, timestamps)
        add_summary_images(field_key, nwb)
