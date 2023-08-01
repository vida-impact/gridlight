#!/usr/bin/env python
# coding: utf-8
import logging
import os
import sys
from datetime import datetime

import click
import geopandas as gpd
import numpy as np
import pandas as pd


from src.gridlight.gridlight import (estimate_mem_use, get_targets_costs,optimise)
from src.gridlight.post import (accuracy, raster_to_lines, thin, threshold_distances)
from src.gridlight.prepare import prepare_roads
from src.gridlight.util.raster import save_2d_array_as_raster

sys.path.append(os.path.abspath("."))
from config import get_config, get_config_1
from src.gridlight.util.remote_storage import (get_default_remote_storage, get_develop_remote_storage)


@click.command()
@click.option(
    "--area-of-interest-data",
    default="nigeria/nigeria-kano.geojson",
    help="Path to AOI vector file relative to data/ground_truth/ directory.",
)
@click.option(
    "--roads-data",
    default="nigeria/nigeria-roads-200101.gpkg",
    help="Path to Roads vector file, relative to data/ground_truth/ directory",
)
@click.option(
    "--population-data",
    default="nigeria/population_nga_2018-10-01.tif",
    help="Path to population raster file, relative to data/ground_truth/ directory",
)
@click.option(
    "--grid-truth-data",
    default="nigeria/nigeriafinal.geojson",
    help="Path to ground truth grid vector file, relative to data/ground_truth/ directory",
)
@click.option(
    "--power-data",
    help="Path to vector file containing power lines, relative to data/processed/ directory",
)
@click.option(
    "--targets-out",
    default="",
    help="Path to directory where the electrification targets have been stored, relative to data/processed/ directory",
)
@click.option(
    "--targets-clean-out",
    default="",
    help="Path to directory where the population filtered electrification targets have been stored, relative to data/processed/ directory",
)
@click.option(
    "--result-subfolder",
    default="nigeria",
    help="Path to directory where clipped nightlight imagery will be stored,"
    "relative to data/processed directory.",
)
@click.option(
    "--dev-mode",
    is_flag=True,
    help="Whether to run the script in dev mode. "
    "If true, the results will be taken from the configured development bucket",
)

def run_gridlight_prediction(
    area_of_interest_data: str,
    roads_data: str,
    grid_truth_data: str,
    power_data: str,
    targets_in: str,
    targets_clean_in:str,
    result_subfolder: str,
    dev_mode: bool,
):
    DEFAULT_CRS = "EPSG:4326"
    c = get_config(reload=True)
    log = logging.getLogger(__name__)
    input_files = {}
        
    if dev_mode:
        remote_storage = get_develop_remote_storage(c)
    else:
        remote_storage = get_default_remote_storage(c)

    input_files["aoi_in"] = c.datafile_path(
        area_of_interest_data,
        stage=c.RAW,
        check_existence=False,
        relative=True,
    )
    input_files["targets_in"] = c.datafile_path(
        f"{targets_in}/targets.tif",
        stage=c.PROCESSED,
        check_existence=False,
        relative=True,
    )
    input_files["targets_clean_in"] = c.datafile_path(
        f"{targets_clean_in}/targets_clean.tif",
        stage=c.PROCESSED,
        check_existence=False,
        relative=True,
    )
    input_files["roads_in"] = c.datafile_path(
        roads_data, stage=c.GROUND_TRUTH, check_existence=False, relative=True
    )

    input_files["grid_truth"] = c.datafile_path(
        grid_truth_data, stage=c.GROUND_TRUTH, check_existence=False, relative=True
    )
    if power_data is not None:
        input_files["power"] = c.datafile_path(
            power_data, stage=c.PROCESSED, check_existence=False, relative=True
        )

    for _, path in input_files.items():
        log.info(f"Pulling file {path} from storage.")
        remote_storage.pull(path, "")

    # Define output paths
    PREDICTIONS_PATH = f"{result_subfolder}/predictions/"
    
    roads_out = c.datafile_path(
        f"{result_subfolder}/roads_clipped.tif",
        stage=c.PROCESSED,
        check_existence=False,
    )

    dist_out = c.datafile_path(
        f"{PREDICTIONS_PATH}/dist.tif", stage=c.PROCESSED, check_existence=False
    )
    guess_out = c.datafile_path(
        f"{PREDICTIONS_PATH}/MV_grid_prediction.tif",
        stage=c.PROCESSED,
        check_existence=False,
    )
    guess_skeletonized_out = c.datafile_path(
        f"{PREDICTIONS_PATH}/MV_grid_prediction_skeleton.tif",
        stage=c.PROCESSED,
        check_existence=False,
    )
    guess_vec_out = c.datafile_path(
        f"{PREDICTIONS_PATH}/MV_grid_prediction",
        stage=c.PROCESSED,
        check_existence=False,
    )

    aoi = gpd.read_file(input_files["aoi_in"])

    params = {
        "cutoff": 0.0
    }

    if power_data is not None:
        power = gpd.read_file(input_files["power"])
        roads_raster, affine = prepare_roads(
            input_files["roads_in"], aoi, input_files["targets_in"], power
        )
    else:
        roads_raster, affine = prepare_roads(input_files["roads_in"], aoi, input_files["targets_in"])
    save_2d_array_as_raster(roads_out, roads_raster, affine, DEFAULT_CRS, nodata=-1)
    log.info(
        f"Costs prepared and saved to {roads_out}, now connecting locations with algorithm."
    )

    targets, costs, start, affine = get_targets_costs(input_files["targets_clean_in"], roads_out)
    est_mem = estimate_mem_use(targets, costs)
    log.info(f"Estimated memory usage of algorithm: {est_mem:.2f} GB")

    dist = optimise(targets, costs, start, jupyter=False, animate=False, affine=affine)
    save_2d_array_as_raster(dist_out, dist, affine, DEFAULT_CRS)

    guess = threshold_distances(dist, threshold=params["cutoff"])
    save_2d_array_as_raster(guess_out, guess, affine, DEFAULT_CRS)
    log.info(f"Prediction is completed and saved to {guess_out}, thinning raster now..")

    guess_skel = thin(guess)
    save_2d_array_as_raster(guess_skeletonized_out, guess_skel, affine, DEFAULT_CRS)
    log.info(
        f"Thinning raster complete and saved to {guess_skeletonized_out}, now converting to vector geometries.."
    )

    guess_gdf = raster_to_lines(guess_skel, affine, DEFAULT_CRS)
    if power_data is not None:
        guess_gdf.append(power)
    guess_gdf.to_file(f"{guess_vec_out}.gpkg", driver="GPKG")
    guess_gdf.to_file(f"{guess_vec_out}.geojson", driver="GeoJSON")

    log.info(
        f"Converted raster to {len(guess_gdf)} grid lines and saved to "
        f"{guess_vec_out}. Evaluating on ground truth now.."
    )

    truth = gpd.read_file(input_files["grid_truth"]).to_crs(DEFAULT_CRS)
    true_pos, false_neg = accuracy(truth, guess_out, aoi)
    log.info(f"Points identified as grid that are grid: {100*true_pos:.0f}%")
    log.info(f"Actual grid that was missed: {100*false_neg:.0f}%")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gridlight_prediction()