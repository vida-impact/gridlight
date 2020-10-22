"""
Test suite for the metrics module
"""
import os
from typing import Optional

import pytest

import rasterio
from affine import Affine
import numpy as np
import geopandas as gp
from shapely.geometry import LineString, Polygon
import rasterio.warp

from gridfinder.metrics import eval_confusion_matrix, ConfusionMatrix


TRANSFORM = Affine(1 + 1e-10, 0.0, 0.0, 0.0, 1 + 1e-10, 0.0)


def store_tif_file(file_path, raster: np.array) -> str:
    with rasterio.open(
        str(file_path),
        "w",
        driver="GTiff",
        height=raster.shape[0],
        width=raster.shape[1],
        count=1,
        dtype=raster.dtype,
        crs="EPSG:3857",
        transform=TRANSFORM,
    ) as dst:
        dst.write(raster, 1)
        dst.close()
    return str(file_path)


CORRECT_GUESS = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)


CORRECT_GUESS_BUT_SHIFTED_LEFT = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)

PARTIALLY_CORRECT_GUESS = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)

PARTIALLY_CORRECT_GUESS_BUT_SHIFTED_LEFT = np.array(
    [
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)


# The value don't map the indices in the grid directly
# The later used rasterize function start counting the indices form one instead of zero
# Also the LineString coordinates need to be transposed
GROUND_TRUTH = gp.GeoDataFrame(
    index=[0], geometry=[LineString([[2, 2], [2, 3], [2, 4], [2, 5]])], crs="EPSG:3857"
)

SAMPLE_AOI = gp.GeoDataFrame(
    index=[0], geometry=[Polygon([[0, 0], [4, 0], [4, 4], [0, 4]])], crs="EPSG:3857"
)


@pytest.fixture()
def ground_truth_lines():
    return GROUND_TRUTH


@pytest.fixture()
def sample_aoi():
    return SAMPLE_AOI


@pytest.fixture()
def correct_guess(raster_saver) -> str:
    return raster_saver(CORRECT_GUESS, "correct_guess.tif")


@pytest.fixture()
def correct_guess_but_shifted_left(raster_saver) -> str:
    return raster_saver(
        CORRECT_GUESS_BUT_SHIFTED_LEFT, "correct_guess_but_shifted_left.tif"
    )


@pytest.fixture()
def partially_correct_guess(raster_saver) -> str:
    return raster_saver(PARTIALLY_CORRECT_GUESS, "partially_correct_guess.tif")


@pytest.fixture()
def partially_correct_guess_but_shifted_left(raster_saver) -> str:
    return raster_saver(
        PARTIALLY_CORRECT_GUESS_BUT_SHIFTED_LEFT,
        "partially_correct_guess_but_shifted_left.tif",
    )


@pytest.fixture()
def raster_saver(tmpdir_factory):
    """ Saves Raster as TIF file"""

    def _raster_saver(raster: np.array, file_name: str):
        tmp_folder = tmpdir_factory.mktemp("tmp_raster")
        path = os.path.join(tmp_folder, file_name)
        store_tif_file(path, raster)
        return rasterio.open(path)

    return _raster_saver


@pytest.mark.parametrize(
    ["raster_guess", "cell_size_in_meters", "expected_confusion_matrix"],
    [
        (
            pytest.lazy_fixture("correct_guess"),
            None,
            ConfusionMatrix(tp=4, fp=0, tn=32, fn=0),
        ),
        (
            pytest.lazy_fixture("correct_guess_but_shifted_left"),
            None,
            ConfusionMatrix(tp=0, fp=4, tn=28, fn=4),
        ),
        (
            pytest.lazy_fixture("partially_correct_guess"),
            None,
            ConfusionMatrix(tp=3, fp=0, tn=32, fn=1),
        ),
        (
            pytest.lazy_fixture("partially_correct_guess_but_shifted_left"),
            None,
            ConfusionMatrix(tp=0, fp=3, tn=29, fn=4),
        ),
        (
            pytest.lazy_fixture("correct_guess"),
            2.0,
            ConfusionMatrix(tp=3, fp=0, tn=6, fn=0),
        ),
        (
            pytest.lazy_fixture("correct_guess_but_shifted_left"),
            2.0,
            ConfusionMatrix(tp=3, fp=0, tn=6, fn=0),
        ),
        (
            pytest.lazy_fixture("partially_correct_guess"),
            2.0,
            ConfusionMatrix(tp=2, fp=0, tn=6, fn=1),
        ),
        (
            pytest.lazy_fixture("partially_correct_guess_but_shifted_left"),
            2.0,
            ConfusionMatrix(tp=2, fp=0, tn=6, fn=1),
        ),
        (
            pytest.lazy_fixture("correct_guess"),
            3.0,
            ConfusionMatrix(tp=2, fp=0, tn=2, fn=0),
        ),
    ],
)
def test_accuracy(
    ground_truth_lines: gp.GeoDataFrame,
    raster_guess: rasterio.DatasetReader,
    cell_size_in_meters: Optional[int],
    expected_confusion_matrix: ConfusionMatrix,
):
    assert (
        eval_confusion_matrix(ground_truth_lines, raster_guess, cell_size_in_meters)[
            "confusion_matrix"
        ]
        == expected_confusion_matrix
    )


def test_accuracy_up_sampling_fails(
    correct_guess: rasterio.DatasetReader, ground_truth_lines: gp.GeoDataFrame
):
    with pytest.raises(ValueError):
        eval_confusion_matrix(ground_truth_lines, correct_guess, 0.5)


@pytest.mark.parametrize(
    ["raster_guess", "cell_size_in_meters", "expected_confusion_matrix"],
    [
        (
            pytest.lazy_fixture("correct_guess"),
            None,
            ConfusionMatrix(tp=3, fp=0, tn=13, fn=0),
        ),
        (
            pytest.lazy_fixture("correct_guess"),
            2.0,
            ConfusionMatrix(tp=2, fp=0, tn=2, fn=0),
        ),
    ],
)
def test_accuracy_with_aoi(
    raster_guess: rasterio.DatasetReader,
    cell_size_in_meters: int,
    sample_aoi: gp.GeoDataFrame,
    ground_truth_lines: gp.GeoDataFrame,
    expected_confusion_matrix: ConfusionMatrix,
):
    assert (
        eval_confusion_matrix(
            ground_truth_lines, raster_guess, cell_size_in_meters, aoi=sample_aoi
        )["confusion_matrix"]
        == expected_confusion_matrix
    )
