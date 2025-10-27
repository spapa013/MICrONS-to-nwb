from typing import Optional

from neuroconv.basedatainterface import BaseDataInterface
from neuroconv.tools.nwb_helpers import make_or_load_nwbfile, get_module
from neuroconv.tools.roiextractors import add_two_photon_series
from neuroconv.utils import FilePathType, get_base_schema, get_schema_from_hdmf_class
from microns_phase3 import nda
from pynwb import NWBFile
from pynwb.ophys import ImagingPlane, TwoPhotonSeries

from .micronstiffimagingextractor import MicronsTiffImagingExtractor


class MicronsTiffImagingInterface(BaseDataInterface):
    """Data interface for adding the 2p calcium imaging to an existing NWB file."""

    def __init__(
        self,
        file_path: FilePathType,
        scan_key: dict,
    ):
        super().__init__(
            file_path=file_path,
            scan_key=scan_key,
        )

    def get_metadata_schema(self):
        metadata_schema = super().get_metadata_schema()

        metadata_schema["required"] = ["Ophys"]
        imaging_plane_schema = get_schema_from_hdmf_class(ImagingPlane)
        two_photon_series_schema = get_schema_from_hdmf_class(TwoPhotonSeries)
        imaging_plane_schema.update(dict(required=["name"]))

        # Initiate Ophys metadata
        metadata_schema["properties"]["Ophys"] = get_base_schema(tag="Ophys")
        metadata_schema["properties"]["Ophys"].update(
            required=["ImagingPlane", "TwoPhotonSeries"],
            properties=dict(
                ImagingPlane=dict(
                    type="array",
                    minItems=1,
                    items=imaging_plane_schema,
                ),
                TwoPhotonSeries=dict(
                    type="array",
                    minItems=1,
                    items=two_photon_series_schema,
                ),
            ),
        )
        return metadata_schema

    def get_metadata(self):
        """Child DataInterface classes should override this to match their metadata."""
        metadata = dict(Ophys=dict(TwoPhotonSeries=[], ImagingPlane=[]))
        all_field_data = (nda.Field() & self.source_data["scan_key"]).fetch(as_dict=True)
        imaging_rate = (nda.Scan() & self.source_data["scan_key"]).fetch1("fps")
        for field_data in all_field_data:
            two_photon_series_name = f"TwoPhotonSeries{field_data['field']}"
            imaging_plane_name = f"ImagingPlane{field_data['field']}"
            dimension = [field_data["px_width"], field_data["px_height"]]
            width_in_meters = field_data["um_width"] / 1e6
            height_in_meters = field_data["um_height"] / 1e6
            imaging_depth = field_data["field_z"] / 1e6

            metadata["Ophys"]["TwoPhotonSeries"].append(
                dict(
                    name=two_photon_series_name,
                    description=f"Calcium imaging data for field {field_data['field']} at {imaging_rate} Hz and {imaging_depth} meters depth.",
                    imaging_plane=imaging_plane_name,
                    dimension=dimension,
                    field_of_view=[width_in_meters, height_in_meters],
                    unit="n.a.",
                )
            )

            metadata["Ophys"]["ImagingPlane"].append(dict(name=imaging_plane_name))

        return metadata

    def run_conversion(
        self,
        nwbfile_path: Optional[str] = None,
        nwbfile: Optional[NWBFile] = None,
        metadata: Optional[dict] = None,
        stub_test: bool = False,
        stub_frames: int = 100,
        overwrite: bool = False,
        verbose: bool = True,
        iterator_type: Optional[str] = "v2",
        iterator_options: Optional[dict] = None,
    ):
        num_frames, num_fields, sampling_frequency = (nda.Scan & self.source_data["scan_key"]).fetch1(
            "nframes", "nfields", "fps"
        )

        with make_or_load_nwbfile(
            nwbfile_path=nwbfile_path,
            nwbfile=nwbfile,
            metadata=metadata,
            overwrite=overwrite,
            verbose=verbose,
        ) as nwbfile_out:
            ophys = get_module(nwbfile_out, "ophys")
            frame_times = (
                ophys.get_data_interface("Fluorescence").roi_response_series["RoiResponseSeries1"].timestamps[:]
            )
            for plane_index in range(num_fields):
                imaging_extractor = MicronsTiffImagingExtractor(
                    file_path=self.source_data["file_path"],
                    sampling_frequency=sampling_frequency,
                    plane_index=plane_index,
                    num_frames_per_plane=num_frames,
                )

                imaging_extractor.set_times(times=frame_times)

                if stub_test:
                    extractor = imaging_extractor.frame_slice(0, stub_frames)

                add_two_photon_series(
                    imaging=extractor if stub_test else imaging_extractor,
                    nwbfile=nwbfile_out,
                    metadata=metadata,
                    two_photon_series_index=plane_index,
                    iterator_type=iterator_type,
                    iterator_options=iterator_options,
                )
                if verbose:
                    print(f"TwoPhotonSeries data for plane {plane_index + 1} is added to nwbfile.")
