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

from src.gridlight.electrificationfilter import NightlightFilter
from src.gridlight.prepare import (drop_zero_pop, merge_rasters, prepare_ntl)
from src.gridlight.util.loading import open_raster_files
from src.gridlight.util.raster import get_clipped_data, save_2d_array_as_raster

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
    "--population-data",
    default="nigeria/population_nga_2018-10-01.tif",
    help="Path to population raster file, relative to data/ground_truth/ directory",
)
@click.option(
    "--nightlight-data",
    default="nightlight_imagery/75N060W",
    help="Path to directory where nightlight imagery should be stored to, relative to data/raw/ directory",
)
@click.option(
    "--start-date",
    type=click.DateTime(formats=["%Y-%m", "%y%m"]),
    default="2021-01",
    help="Start date of the range defining the nightlight files which shall be used. Note: If you want to include the first month, type YYYY-MM-01.",
)
@click.option(
    "--end-date",
    type=click.DateTime(formats=["%Y-%m", "%y%m"]),
    default="2021-12",
    help="End date of the range defining the nightlight files which shall be used. Note: The month you specfy here will be included, irrespective of the date.",
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
def run_gridtargets(
    area_of_interest_data: str,
    population_data: str,
    nightlight_data: str,
    start_date: datetime.date,
    end_date: datetime.date,
    result_subfolder: str,
    dev_mode: bool,
):
    DEFAULT_CRS = "EPSG:4326"
    c = get_config(reload=True)
    d = get_config_1(reload=True)
    log = logging.getLogger(__name__)
    input_files = {}
    

    ntl_monthly_dates = pd.date_range(start=start_date, end=end_date, freq="MS")
    all_ntl_input_monthly_filenames = [
        f"{date.strftime('%Y')}/{date.strftime('%Y%m%d')}.tif" for date in ntl_monthly_dates
    ]

    all_ntl_input_monthly_full_paths = [
        d.datafile_path(
            f"{nightlight_data}/{ntl_filename}",
            stage=c.RAW,
            check_existence=False,
            relative=True,
        )
        for ntl_filename in all_ntl_input_monthly_filenames
    ]

    if dev_mode:
        remote_storage_1 = get_develop_remote_storage(d)
    else:
        remote_storage_1 = get_default_remote_storage(d)

    for path in all_ntl_input_monthly_full_paths:
        log.info(f"Pulling nightlight imagery file {path} from storage.")
        remote_storage_1.pull(path, "")
    
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
    input_files["pop_in"] = c.datafile_path(
        population_data, stage=c.GROUND_TRUTH, check_existence=False, relative=True
    )

    for _, path in input_files.items():
        log.info(f"Pulling file {path} from storage.")
        remote_storage.pull(path, "")

    # Define output paths
    ELECTRIFICATION_TARGET_PATH = f"{result_subfolder}/electrification_targets"
    CLIPPED_NTL_PATH = f"{result_subfolder}/ntl_clipped"
    folder_ntl_out = c.datafile_path(
        CLIPPED_NTL_PATH, stage=c.PROCESSED, check_existence=False
    )
    raster_merged_out = c.datafile_path(
        f"{result_subfolder}/ntl_merged.tif", stage=c.PROCESSED, check_existence=False
    )
    targets_out = c.datafile_path(
        f"{ELECTRIFICATION_TARGET_PATH}/targets.tif",
        stage=c.PROCESSED,
        check_existence=False,
    )
    targets_clean_out = c.datafile_path(
        f"{ELECTRIFICATION_TARGET_PATH}/targets_clean.tif",
        stage=c.CLEANED,
        check_existence=False,
    )

    params = {
        "percentile": 70,
        "ntl_threshold": 0.1,
        "upsample_by": 2,
        "cutoff": 0.0,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }

    aoi = gpd.read_file(input_files["aoi_in"])

    log.info(f"Preprocess raw nightlight images.")

    output_paths = []
    for ntl_file, full_path in zip(
        all_ntl_input_monthly_filenames, all_ntl_input_monthly_full_paths
    ):
        output_paths.append(
            os.path.join(folder_ntl_out, f"{ntl_file}")
        )  # stripping off the .tgz
        with open_raster_files(full_path) as raster:
            clipped_data, transform = get_clipped_data(raster, aoi, nodata=np.nan)
            save_2d_array_as_raster(
                output_paths[-1], clipped_data, transform, crs=raster.crs.to_string()
            )
            log.info(f"Stored {full_path} as 2d raster in {output_paths[-1]}.")

    raster_merged, affine, _ = merge_rasters(
        output_paths, percentile=params["percentile"]
    )
    save_2d_array_as_raster(raster_merged_out, raster_merged, affine, DEFAULT_CRS)
    log.info(
        f"Merged {len(raster_merged)} rasters of nightlight imagery to {raster_merged_out}"
    )

    ntl_filter = NightlightFilter()

    ntl_thresh, affine = prepare_ntl(
        raster_merged,
        affine,
        electrification_predictor=ntl_filter,
        threshold=params["ntl_threshold"],
        upsample_by=params["upsample_by"],
    )

    save_2d_array_as_raster(targets_out, ntl_thresh, affine, DEFAULT_CRS)
    log.info(f"Targets prepared from nightlight imagery and saved to {targets_out}.")

    targets_clean = drop_zero_pop(targets_out, input_files["pop_in"], aoi)
    save_2d_array_as_raster(targets_clean_out, targets_clean, affine, DEFAULT_CRS)
    log.info(f"Removed locations with zero population and saved to {targets_clean_out}")

    remote_storage_1.push(targets_out, "")
    remote_storage_1.push(targets_clean_out, "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_gridtargets()