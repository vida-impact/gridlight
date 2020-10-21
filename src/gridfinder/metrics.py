""" Metrics module implements calculation of confusion matrix given a prediction """
from typing import Optional, Tuple

import fiona
from dataclasses import dataclass

from affine import Affine
import numpy as np
import geopandas as gp
from sklearn.metrics import confusion_matrix
import rasterio
from rasterio.enums import Resampling
import rasterio.warp
from rasterio.warp import reproject
from rasterio.features import rasterize

from gridfinder._util import clip_line_poly
from gridfinder.util.raster import get_clipped_data, get_resolution_in_meters


@dataclass()
class ConfusionMatrix:
    tp: float = 0.0
    fp: float = 0.0
    tn: float = 0.0
    fn: float = 0.0


def eval_confusion_matrix(
    ground_truth_lines: gp.GeoDataFrame,
    raster_guess_reader: rasterio.DatasetReader,
    cell_size_in_meters: Optional[float] = None,
    aoi: Optional[gp.GeoDataFrame] = None,
):
    """
    Calculates the
     - true positives
     - true negatives
     - false positives
     - false negatives
    of a grid line prediction based the provided ground truth.

    :param ground_truth_lines: A gp.GeoDataFrame object which contains LineString objects as shapes
                               representing the grid lines.
    :param raster_guess_reader: A rasterio.DatasetReader object which contains the raster of predicted grid lines.
                                Pixel values marked with 1 are considered a prediction of a grid line.
    :param cell_size_in_meters: The cell_size_in_meters parameter controls the size of one prediction in meters.
                                E.g. the original raster has a pixel size of 100m x 100m.
                                A cell_size of 1000m meters means that one prediction
                                is now the grouping of 100 original pixels.
                                This is done for both the ground truth raster and the prediction raster.
                                The down-sampling strategy considers a collection of pixel values as a positive
                                prediction (value = 1) if at least one pixel in that collection has the value 1.
    :param aoi: A gp.GeoDataFrame containing exactly one Polygon or Multipolygon marking the area of interest.
                The CRS is expected to be the same as the raster_guess_readers' CRS.

    :returns: ConfusionMatrix

    """
    def perform_scaling(
        raster_array: np.array, affine_mat: Affine, scaling_factor: float, crs: str
    ) -> Tuple[np.array, Affine]:
        shape = (
            1,
            round(raster_array.shape[0] * scaling_factor),
            round(raster_array.shape[1] * scaling_factor),
        )
        raster_out = np.empty(shape)

        raster_out_transform = affine_mat * affine_mat.scale(
            (raster_array.shape[0] / shape[0]), (raster_array.shape[1] / shape[1])
        )
        with fiona.Env():
            with rasterio.Env():
                reproject(
                    raster_array,
                    raster_out,
                    src_transform=affine_mat,
                    dst_transform=raster_out_transform,
                    src_crs={"init": crs},
                    dst_crs={"init": crs},
                    resampling=Resampling.max,
                )
        return raster_out.squeeze(axis=0), raster_out_transform

    def rasterize_geo_dataframe(
        raster_array: np.array, data_frame: gp.GeoDataFrame, transform: Affine
    ) -> np.array:
        """ All raster values where shapes are found will have the values one, the rest zero."""
        assert (
            len(raster_array.shape) == 2
        ), f"Expected 2D array, got shape {raster_array.shape}."
        data_rows = [row.geometry for _, row in data_frame.iterrows()]
        new_raster = rasterize(
            data_rows,
            out_shape=raster_array.shape,
            fill=0,
            default_value=1,
            all_touched=True,
            transform=transform,
        )
        return new_raster

    # perform clipping of raster and ground truth in case aoi parameter is provided
    if aoi is not None:
        ground_truth_lines = clip_line_poly(ground_truth_lines, aoi)
        raster, affine = get_clipped_data(raster_guess_reader, aoi)
        raster = raster.squeeze(axis=0)
    else:
        raster, affine = raster_guess_reader.read(1), raster_guess_reader.transform

    # perform down-sampling in case cell_size_in_meters parameter is provided.
    if cell_size_in_meters is not None:
        current_cell_size_x, current_cell_size_y = get_resolution_in_meters(
            raster_guess_reader
        )
        if current_cell_size_x != current_cell_size_y:
            raise ValueError(
                f"Only quadratic pixel values are supported for scaling."
                f" Found pixel size x {current_cell_size_x}"
                f" and pixel size y P{current_cell_size_y}."
            )
        scaling = current_cell_size_x / cell_size_in_meters
        if scaling > 1.0:
            raise ValueError(
                f"Up-sampling not supported. Select a cell size of at least {current_cell_size_x}."
            )
        raster, affine = perform_scaling(
            raster, affine, scaling, crs=raster_guess_reader.crs.to_string()
        )

    raster_ground_truth = rasterize_geo_dataframe(raster, ground_truth_lines, affine)
    mat = confusion_matrix(raster_ground_truth.flatten(), raster.flatten())

    return ConfusionMatrix(tp=mat[1, 1], fp=mat[0, 1], fn=mat[1, 0], tn=mat[0, 0])
