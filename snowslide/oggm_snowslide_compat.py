"""This module allows the use of snowslide within the OGGM workflow
"""

# Module logger
import logging
log = logging.getLogger(__name__)

# Built ins
import logging
import os
from time import gmtime, strftime
import re as regexp

# External libs
import cftime
import numpy as np
import xarray as xr
import pandas as pd

# Locals
from oggm import __version__
import oggm.cfg as cfg
from oggm import utils
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.exceptions import InvalidWorkflowError, InvalidParamsError
from oggm.core.massbalance import MassBalanceModel, mb_calibration_from_scalar_mb, decide_winter_precip_factor, mb_calibration_from_geodetic_mb, MonthlyTIModel
from snowslide.snowslide_main import snowslide_base
from oggm.utils import get_temp_bias_dataframe, clip_scalar
from oggm.core import massbalance
from oggm.core.flowline import flowline_from_dataset, glacier_from_netcdf, FileModel, MixedBedFlowline

# Climate relevant global params
MB_GLOBAL_PARAMS = [
    "temp_default_gradient",
    "temp_all_solid",
    "temp_all_liq",
    "temp_melt",
]


@utils.entity_task(log, writes=["gridded_data"])
def snowslide_to_gdir(gdir, routing="mfd", Propagation=True, snd0=None):
    """Add an idealized estimation of avalanches snow redistribution to this glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    snd0 : float
        initial snow depth (in meters) chosen to run the Snowslide simulation
    """
    # Get the path to the gridded data file and open it
    gridded_data_path = gdir.get_filepath("gridded_data")
    with xr.open_dataset(gridded_data_path) as ds:
        ds = ds.load()

    # Get the path of the dem and climate data
    path_to_dem = gdir.get_filepath("dem")

    # if initial snow depth not given, use default value of 1m snow everywhere
    if snd0 is None:
        snd0 = np.ones_like(ds.topo.data)
    
    # Launch snowslide simulation with initial snow depth
    param_routing = {"routing": routing, "preprocessing": True, "compute_edges": True}
    snd = snowslide_base(
        path_to_dem,
        snd0=snd0,
        param_routing=param_routing,
        glacier_id=f"({gdir.rgi_id}) ",
        propagation_boolean=Propagation
    )

    # Write
    with utils.ncDataset(gdir.get_filepath("gridded_data"), "a") as nc:
        vn = "snowslide_1m"

        # delete the variable if it already exists
        if vn in nc.variables:
            nc.remove_variable(vn)

        # create the variable
        v = nc.createVariable(vn, "f4", ("y", "x"), zlib=True)

        # set attributes
        v.units = "m"
        v.long_name = "Snowcover after avalanches"
        # assign data to variable
        v[:] = snd


@utils.entity_task(log, writes=["gridded_data"])
def snowslide_to_gdir_meanmonthly(gdir, clim_path='climate_historical', routing="mfd", Propagation=True, default_grad=-0.0065, t_solid=0, t_liq=2, 
    ys=2000, ye=2020, rho_freshsnow=200, climate_input_filesuffix='', store_snd_before=False):
    """Add an estimation of avalanches snow redistribution to this glacier directory for a given period
    using given climate data using a mean monthly aggregation

    Parameters
    ----------
    gdir: :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    clim_path: str
        path to climate data withing gdir (either 'climate_historical' or 'gcm_data' with a given file_suffix)
    default_grad: np.float
        temperature gradient as a function of elevation (K/m)
    t_solid: np.float
        temperature below which all the precipitation is solid (°C)
    t_liq: np.float
        temperature above which all the precipitation is liquid (°C)
    ys: np.float
        Start year for SnowSlide simulation
    ye: np.float
        End year for SnowSlide simulation
    rho_freshsnow: np.float
        Density of fresh snow (kg/m3) to convert to snow height (input precipitation data is in kg/m2)
    climate_input_filesuffix: str
        filesuffix for the input climate file
    store_snd_before: bool
        argument to store snow accumulation before avalanching. Only useful to calculate volume of snow removed and added by avalanches
    """
    # number of years
    n_years = ye-ys
    
    # Read climate data
    fpath = gdir.get_filepath(clim_path, filesuffix=climate_input_filesuffix)
    with xr.open_dataset(fpath) as ds:
        ds = ds.load()

    # monthly temperature and precipitation the wanted period (01/20XX-01/20YY)
    temp = ds.temp.sel(time=slice(f"{ys}-01", f"{ye}-01")).values
    prcp = ds.prcp.sel(time=slice(f"{ys}-01", f"{ye}-01")).values

    grad = prcp * 0 + default_grad
    ref_hgt = ds.ref_hgt

    # Get minimum and maximum altitude in dem (to compute snow height from precipitation)
    gridded_data_path = gdir.get_filepath("gridded_data")
    with xr.open_dataset(gridded_data_path) as ds:
        ds = ds.load()
    
    # snowslide parameters
    param_routing = {"routing": routing, "preprocessing": True, "compute_edges": True}

    # Get the path of the dem and climate data
    path_to_dem = gdir.get_filepath("dem")

    ## run snowslide MEAN MONTHLY over the full time period to compute yearly Prcp fact as a function of altitude
    snd_before = xr.zeros_like(ds['topo'])
    snd_after = xr.zeros_like(ds['topo'])

    for ii in range(0, 12):
        sndm = np.zeros_like(ds.topo.data)
        for jj in range(0,n_years):
            # calculate temp & prcp as a function of elevation
            itemp = temp[ii+jj*12]
            iprcp = prcp[ii+jj*12]
            igrad = grad[ii+jj*12]

            # compute elevation of liquid and solid precip
            zliq = ref_hgt + (t_liq-itemp)/igrad
            zsolid = ref_hgt + (t_solid-itemp)/igrad

            # Compute initial snow depth
            snd0 = np.ones_like(ds.topo.data)
            snd0[ds.topo.data>zsolid] = iprcp
            snd0[ds.topo.data<zliq] = 0
            heights_mix = ds.topo.data[(ds.topo.data<=zsolid) & (ds.topo.data>=zliq)]
            snd0[(ds.topo.data<=zsolid) & (ds.topo.data>=zliq)] = iprcp * 1 - (itemp + igrad * (heights_mix - ref_hgt) - t_solid) / (t_liq - t_solid)

            # Convert to snow height (density conversion from kg/m2)
            snd0 = snd0/rho_freshsnow
            
            # add to monthly snow depth
            sndm = sndm+snd0

        sndm = sndm/n_years

        # Run snowslide
        snd = snowslide_base(
            path_to_dem,
            snd0=sndm,
            param_routing=param_routing,
            glacier_id=f"({gdir.rgi_id}) ",
            propagation_boolean=Propagation
        )

        # allocate 
        snd_after = snd_after+snd
        snd_before = snd_before+sndm

    # pfact from snd_after & snd_before
    pfact = snd_after/snd_before

    # Write to get gridded snowline_1m product
    with utils.ncDataset(gdir.get_filepath("gridded_data"), "a") as nc:
        vn = "snowslide_1m"

        # create the variable if it does not already exists
        if vn in nc.variables:
            v = nc.variables[vn]
        else:
            # create the variable
            v = nc.createVariable(vn, "f4", ("y", "x"), zlib=True)
            # set attributes
            v.units = "m"
            v.long_name = "Snowcover after avalanches"

        # assign data to variable
        v[:] = pfact

        if store_snd_before:
            vn_b = "snowslide_snd_before"
            if vn_b in nc.variables:
                vb = nc.variables[vn_b]
            else:
                vb = nc.createVariable(vn_b, "f4", ("y", "x"), zlib=True)
                vb.units = "m"
                vb.long_name = "mean annual snow height before redistribution"
            vb[:] = snd_before


@utils.entity_task(log, writes=['gridded_simulation'])
def update_topo(gdir, input_filesuffix='', output_filesuffix=''):
    """Add the avalanche contribution and altitude distribution with time to the glacier directory 

    Parameters
    ----------
    gdir: :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    input_filesuffix: str
        string ID for GCM, SSP, control/avalanche scenario. 
    output_filesuffix: str
        output file suffix
    """

    with xr.open_dataset(gdir.get_filepath('gridded_simulation', filesuffix=input_filesuffix)) as ds:
        ds = ds.load()

    with xr.open_dataset(gdir.get_filepath('gridded_data')) as gridded_data:
        gridded_data = gridded_data.load()
  
    # select last year of simulation to update topography
    year = ds.time[-1]

    # Select data for the current year and fill NaNs with zeros
    thickness_year = ds.simulated_thickness.sel(time=year).fillna(0)
    
    # Compute updated topography based on the current year's thickness
    updated_topo = ds.bedrock + thickness_year

    # assign new topo to old topo in gridded_data
    gridded_data = gridded_data.assign(topo=updated_topo)

    # save back
    gridded_data.to_netcdf(gdir.get_filepath('gridded_data',
                                   filesuffix=output_filesuffix))


# @utils.entity_task(log, writes=['gridded_simulation'])
# def avalanche_contribution_evo(gdir, input_filesuffix='', output_filesuffix=''):
#     """Add the avalanche contribution and altitude distribution with time to the glacier directory 
# 
#     Parameters
#     ----------
#     gdir: :py:class:`oggm.GlacierDirectory`
#         the glacier directory to process
#     input_filesuffix: str
#         string ID for GCM, SSP, control/avalanche scenario. 
#     output_filesuffix: str
#         output file suffix
#     """
# 
#     with xr.open_dataset(gdir.get_filepath('gridded_simulation', filesuffix=input_filesuffix)) as ds:
#         ds = ds.load()
# 
#     with xr.open_dataset(gdir.get_filepath('gridded_data')) as gridded_data:
#         gridded_data = gridded_data.load()
# 
#     # Initialize lists to store computed values
#     mean_snowslide = [] if 'snowslide_1m' in gridded_data else None
#     median_topo = []
#     min_topo = []
#     max_topo = []
#     
#     # Loop over each year
#     for year in ds["time"]:
#         # Select data for the current year
#         thickness_year = ds.simulated_thickness.sel(time=year)
#         
#         # Create a mask where simulated thickness > 0
#         mask = thickness_year > 0
#         
#         if mean_snowslide is not None:
#             # Mask snowslide_1m based on the current year's thickness
#             snowslide_masked = gridded_data.snowslide_1m.where(mask)
#             
#             # Compute statistics (ignoring NaNs)
#             mean_value_snowslide = snowslide_masked.mean().item()
#             mean_snowslide.append(mean_value_snowslide)
#         
#         # Compute updated topography based on the current year's thickness
#         updated_topo = ds.bedrock + thickness_year
#         topo_masked = updated_topo.where(mask)
#         
#         median_value_topo = topo_masked.median().item()
#         min_value_topo = topo_masked.min().item()
#         max_value_topo = topo_masked.max().item()
#         
#         median_topo.append(median_value_topo)
#         min_topo.append(min_value_topo)
#         max_topo.append(max_value_topo)
# 
#     # Convert lists to xarray DataArrays
#     if mean_snowslide is not None:
#         ds["mean_snowslide"] = ("time", mean_snowslide)
#     ds["median_topo"] = ("time", median_topo)
#     ds["min_topo"] = ("time", min_topo)
#     ds["max_topo"] = ("time", max_topo)
# 
#     ds.to_netcdf(gdir.get_filepath('gridded_simulation',
#                                    filesuffix=output_filesuffix))


def _fallback(gdir):
    """If something wrong happens below"""
    d = dict()
    # Easy stats - this should always be possible
    d["rgi_id"] = gdir.rgi_id
    return d


@utils.entity_task(log, writes=['gridded_simulation'])
def avalanche_contribution_evo(gdir, input_filesuffix=''):
    """Add the avalanche contribution and altitude distribution with time to the glacier directory 

    Parameters
    ----------
    gdir: :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    input_filesuffix: str
        string ID for GCM, SSP, control/avalanche scenario. 
    output_filesuffix: str
        output file suffix
    """

    with xr.open_dataset(gdir.get_filepath('fl_diagnostics', filesuffix=input_filesuffix), group='fl_0') as ds:
        fl = ds.load()

    with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix=input_filesuffix)) as ds:
        model_diag = ds.load()

    # get binned pfact
    binned_data_file = gdir.get_filepath('elevation_band_flowline', filesuffix='_fixed_dx')
    binned_data = pd.read_csv(binned_data_file, index_col=0)
    snowslide_1D = binned_data.snowslide_1m

    # calculate mean snowslide as a function of time

    # Extract the distance values
    dis_along_flowline = fl.dis_along_flowline.values

    # Create a full snowslide_1D array with ones where missing
    snowslide_extended = pd.Series(np.ones(len(dis_along_flowline)), index=dis_along_flowline)

    # Assign existing values where available
    snowslide_extended.update(snowslide_1D)

    # Convert snowslide_extended to an xarray DataArray
    snowslide_da = xr.DataArray(snowslide_extended.values, coords=[dis_along_flowline], dims=["dis_along_flowline"])

    # Compute the weighted mean for each time step
    mean_snowslide = (snowslide_da * fl.area_m2).sum(dim="dis_along_flowline") / fl.area_m2.sum(dim="dis_along_flowline")

    # Add mean_snowslide to the dataset
    model_diag["mean_snowslide"] = mean_snowslide

    # Save back to the same file, overwriting it while preserving all variables
    model_diag.to_netcdf(gdir.get_filepath('model_diagnostics', filesuffix=input_filesuffix), mode="w")

# @utils.entity_task(log, fallback=_fallback)
# def distributed_outputs(gdir, input_filesuffix='', GCM='', SCENARIO='', TEST=''):
#     """Gather values for glacier-wide avalanche contribution and glacier altitudinal distribution with time
# 
#     Parameters
#     ----------
#     gdir: :py:class:`oggm.GlacierDirectory`
#         the glacier directory to process
#     input_filesuffix: str
#         string ID for GCM, SSP, control/avalanche scenario. 
#     GCM: str
#         string for GCM scenario
#     SCENARIO: str
#         string for ssp scenario
#     TEST: str
#         string for test scenario
#     """
# 
#     # Base dictionary for metadata
#     meta_data = {
#         "rgi_id": gdir.rgi_id,
#         "GCM": GCM,
#         "SSP": SCENARIO,
#         "TEST": TEST,
#     }
# 
#     variables = ["mean_snowslide", "min_topo", "max_topo", "median_topo"]
#     time_range = np.arange(2000, 2102, 1)
# 
#     # Dictionary to store data for each variable
#     data_dicts = {}
# 
#     try:
#         with xr.open_dataset(gdir.get_filepath('gridded_simulation', filesuffix=f'{input_filesuffix}_gridded')) as ds:
#             time_in_gdir = ds["time"].data
#             
#             for var in variables:
#                 if var in ds:  # Check if variable exists
#                     var_dict = meta_data.copy()  # Copy metadata for each variable
#                     data = ds[var].data
#                     
#                     for t in time_range:
#                         mask = time_in_gdir == t
#                         var_dict[t] = np.mean(data[mask]) if np.any(mask) else np.nan  # Avoid empty slices
#                     
#                     data_dicts[var] = var_dict  # Store dictionary for this variable
# 
#         d1 = data_dicts["mean_snowslide"]
#         d2 = data_dicts["min_topo"]
#         d3 = data_dicts["max_topo"]
#         d4 = data_dicts["median_topo"]
# 
#         # Convert each dictionary to a DataFrame (keys become the index, values become a column)
#         d1 = pd.DataFrame.from_dict(d1, orient="index", columns=["mean_snowslide"])
#         d2 = pd.DataFrame.from_dict(d2, orient="index", columns=["min_topo"])
#         d3 = pd.DataFrame.from_dict(d3, orient="index", columns=["max_topo"])
#         d4 = pd.DataFrame.from_dict(d4, orient="index", columns=["median_topo"])
#     except Exception as e:
#         print(f"Error processing {input_filesuffix}: {e}")
# 
#     return d1, d2, d3, d4

# @utils.global_task(log)
# def compile_distributed_outputs(gdirs, output_filesuffix="", input_filesuffix='', GCM='', SCENARIO='', TEST='', dir_path=None):
#     """Gather as much statistics as possible about the distributed outputs
# 
#     It can be used to do result diagnostics and other stuffs.
# 
#     Parameters
#     ----------
#     gdirs : list of :py:class:`oggm.GlacierDirectory` objects
#         the glacier directories to process
#     output_filesuffix : str
#         add suffix to output file
#     path : str
#         Folder where to write the csv file. Defaults to cfg.PATHS["working_dir"]
#     input_filesuffix: str
#         string ID for GCM, SSP, control/avalanche scenario. 
#     GCM: str
#         string for GCM scenario
#     SCENARIO: str
#         string for ssp scenario
#     TEST: str
#         string for test scenario
#     """
#     from oggm.workflow import execute_entity_task
# 
#     try:
#         out_df = execute_entity_task(distributed_outputs,gdirs, 
#                                     input_filesuffix=input_filesuffix, GCM=GCM, SCENARIO=SCENARIO, TEST=TEST)
# 
#         d1, d2, d3, d4 = out_df[1]  # out_df is a tuple of dictionaries
# 
#         # Convert outputs to lists
#         mean_snowslide = list(d1.values())
#         min_topo = list(d2.values())
#         max_topo = list(d3.values())
#         median_topo = list(d4.values())
# 
#         if dir_path is None:
#             dir_path = cfg.PATHS["working_dir"]
# 
#         # Define output paths
#         all_results = {
#             "mean_snowslide": [],
#             "min_topo": [],
#             "max_topo": [],
#             "median_topo": [],
#         }
# 
#         # append results
#         all_results["mean_snowslide"].append(mean_snowslide)
#         all_results["min_topo"].append(min_topo)
#         all_results["max_topo"].append(max_topo)
#         all_results["median_topo"].append(median_topo)
# 
#         output_files = {
#             "mean_snowslide": f"glacier_avalanche_contribution_vs_time{input_filesuffix}.csv",
#             "min_topo": f"min_glacier_elev_vs_time{input_filesuffix}.csv",
#             "max_topo": f"max_glacier_elev_vs_time{input_filesuffix}.csv",
#             "median_topo": f"median_glacier_elev_vs_time{input_filesuffix}.csv",
#         }
# 
#         # Extract the common indices from the first dictionary
#         indices = list(d1.keys())
# 
#         # Concatenate results and save to CSV
#         for key, filename in output_files.items():
#             output_path = os.path.join(dir_path, filename)
#             if all_results[key]:  # Avoid writing empty files
#                 # Construct DataFrame with indices in the first row
#                 df = pd.DataFrame([indices] + all_results[key])
#                 df.to_csv(output_path, index=False, header=False)
#                 print(f"Saved {key} to {output_path}")
#             else:
#                 print(f"Warning: No data for {key}, skipping file save.")
#     except Exception as e:
#         print(f"Error processing {input_filesuffix}: {e}")
# 
#     return df


@utils.entity_task(log, fallback=_fallback)
def snowslide_statistics(gdir):
    """Gather statistics about the Snowslide snow redistribution"""

    try:
        # This is because the fallback doesnt have a return value (yet)
        resolution = abs(gdir.grid.dx)
    except:
        resolution = np.NaN

    d = dict()
    # Easy stats - this should always be possible
    d["rgi_id"] = gdir.rgi_id
    d["rgi_region"] = gdir.rgi_region
    d["rgi_subregion"] = gdir.rgi_subregion
    d["rgi_area_km2"] = gdir.rgi_area_km2
    d["map_dx"] = resolution
    d["cenlat"] = gdir.cenlat
    d["cenlon"] = gdir.cenlon
    d["snowslide_1m_glacier_average"] = np.NaN
    d["snowslide_deposit_area_km2"] = np.NaN
    d["snowslide_total_accumulation_km3"] = np.NaN
    d["snowslide_volume_added_km3"] = np.NaN
    d["snowslide_volume_removed_km3"] = np.NaN
    d["melt_f"] = gdir.read_json("mb_calib")['melt_f']
    d["temp_bias"] = gdir.read_json("mb_calib")['temp_bias']
    d["prcp_fac"] = gdir.read_json("mb_calib")['prcp_fac']
    d["melt_f_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['melt_f']
    d["temp_bias_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['temp_bias']
    d["prcp_fac_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['prcp_fac']

    try:
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            mask = ds["glacier_mask"]
            result = ds["snowslide_1m"].where(mask, np.NaN)
            d["snowslide_1m_glacier_average"] = result.mean().data
            d["snowslide_deposit_area_km2"] = (
                float(result.where(result > 1, drop=True).count())
                * resolution**2 * 1e-6
            )
            if "snowslide_snd_before" in ds.variables:
                snd_before = ds["snowslide_snd_before"].where(mask, np.NaN)
                diff = result * snd_before - snd_before

                d["snowslide_total_accumulation_km3"] = (
                    float((result * snd_before).sum()) * resolution**2 * 1e-9
                )
                d["snowslide_volume_added_km3"] = (
                    float(diff.where(diff > 0, drop=True).sum()) * resolution**2 * 1e-9
                )
                d["snowslide_volume_removed_km3"] = (
                    float(diff.where(diff < 0, drop=True).sum()) * resolution**2 * 1e-9
                )
    except (FileNotFoundError, AttributeError, KeyError):
        pass

    return d


@utils.global_task(log)
def compile_snowslide_statistics(gdirs, filesuffix="", dir_path=None):
    """Gather as much statistics as possible about a list of glaciers.

    It can be used to do result diagnostics and other stuffs.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str
        Folder where to write the csv file. Defaults to cfg.PATHS["working_dir"]
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(snowslide_statistics, gdirs)

    out = pd.DataFrame(out_df).set_index("rgi_id")

    if dir_path is None:
        dir_path = cfg.PATHS["working_dir"]

    out_file = os.path.join(dir_path, f"snowslide_statistics{filesuffix}.csv")
    out.to_csv(out_file)

    return out


@utils.entity_task(log, fallback=_fallback)
def binned_statistics(gdir):
    """Binned snowslide stats.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        where to write the data
    """

    d1 = dict()
    d2 = dict()
    d3 = dict() # SMB control
    d4 = dict() # SMB ava

    ## avalanche correction factor & area per elevation bin
    # Easy stats - this should always be possible
    d1["rgi_id"] = gdir.rgi_id
    d2["rgi_id"] = gdir.rgi_id
    d3["rgi_id"] = gdir.rgi_id
    d4["rgi_id"] = gdir.rgi_id

    try:
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            dem = ds["topo"].data
            valid_mask = ds["glacier_mask"].data
            avalanche = ds["snowslide_1m"].data
        
        # Get flowline
        flowline = gdir.read_pickle('inversion_flowlines')[0]
        # get mass balance model
        mb_control = MonthlyTIModel(gdir)
        mb_ava = MonthlyTIAvalancheModel(gdir)
    except:
        return d1, d2, d3, d4

    bsize = 50.0
    dem_on_ice = dem[valid_mask == 1]
    avalanche_on_ice = avalanche[valid_mask == 1]

    bins = np.arange(
        utils.nicenumber(dem_on_ice.min(), bsize, lower=True),
        utils.nicenumber(dem_on_ice.max(), bsize) + 0.01,
        bsize,
    )

    topo_digi = np.digitize(dem_on_ice, bins) - 1
    
    ## SMB 2000-2020 (control & ava models)

    # get annual mass balance at bin elevation
    df_control = pd.DataFrame(index=flowline.dx_meter * np.arange(flowline.nx))
    df_ava = pd.DataFrame(index=flowline.dx_meter * np.arange(flowline.nx))
    for year in range(2000, 2020):
        df_control[year] = mb_control.get_annual_mb(flowline.surface_h, year=year) * cfg.SEC_IN_YEAR * mb_control.rho
        df_ava[year] = mb_ava.get_annual_mb(flowline.surface_h, year=year) * cfg.SEC_IN_YEAR * mb_control.rho

    df_control = df_control.mean(axis=1)
    df_ava = df_ava.mean(axis=1)

    # linearly interpolate to bin elevation
    df_ava_interpolated = np.interp(bins, np.flip(flowline.surface_h), np.flip(df_ava.values)) # need to flip the arrays for elevation to be monotonically increasing
    df_control_interpolated = np.interp(bins, np.flip(flowline.surface_h), np.flip(df_control.values))

    df_ava_interpolated

    for b, bs in enumerate((bins[1:] + bins[:-1]) / 2):
        on_bin = topo_digi == b
        d1["{}".format(np.round(bs).astype(int))] = np.mean(avalanche_on_ice[on_bin])
        d2["{}".format(np.round(bs).astype(int))] = np.sum(on_bin) * gdir.grid.dx**2
        d3["{}".format(np.round(bs).astype(int))] = df_control_interpolated[b]
        d4["{}".format(np.round(bs).astype(int))] = df_ava_interpolated[b]

    return d1, d2, d3, d4


@utils.global_task(log)
def compile_binned_statistics(gdirs, filesuffix="", dir_path=None):
    """Gather statistics about dems on binned elevations.

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    filesuffix : str
        add suffix to output file
    path : str
        Folder where to write the csv file. Defaults to cfg.PATHS["working_dir"]
    """
    from oggm.workflow import execute_entity_task

    out_df = execute_entity_task(binned_statistics, gdirs)

    ava = pd.DataFrame([d[0] for d in out_df]).set_index("rgi_id")
    area = pd.DataFrame([d[1] for d in out_df]).set_index("rgi_id")
    mb_control = pd.DataFrame([d[2] for d in out_df]).set_index("rgi_id")
    mb_ava = pd.DataFrame([d[3] for d in out_df]).set_index("rgi_id")

    ava = ava[sorted(ava.columns)]
    area = area[sorted(area.columns)]
    mb_control = mb_control[sorted(mb_control.columns)]
    mb_ava = mb_ava[sorted(mb_ava.columns)]

    if dir_path is None:
        dir_path = cfg.PATHS["working_dir"]

    out_file = os.path.join(dir_path, f"binned_avalanche_statistics{filesuffix}.csv")
    ava.to_csv(out_file)

    out_file = os.path.join(dir_path, f"binned_area{filesuffix}.csv")
    area.to_csv(out_file)

    out_file = os.path.join(dir_path, f"binned_mb_control{filesuffix}.csv")
    mb_control.to_csv(out_file)

    out_file = os.path.join(dir_path, f"binned_mb_ava{filesuffix}.csv")
    mb_ava.to_csv(out_file)

    return ava, area, mb_control, mb_ava


class MonthlyTIAvalancheModel(MassBalanceModel):
    """Monthly temperature index model."""

    def __init__(
        self,
        gdir,
        filename="climate_historical",
        input_filesuffix="",
        fl_id=None,
        melt_f=None,
        temp_bias=None,
        prcp_fac=None,
        bias=0,
        ys=None,
        ye=None,
        repeat=False,
        check_calib_params=True,
        params_filesuffix='_with_ava',
    ):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data. Default is 'climate_historical'
        input_filesuffix : str, optional
            append a suffix to the filename (useful for GCM runs).
        fl_id : int, optional
            if this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            set to the value of the melt factor you want to use,
            here the unit is kg m-2 day-1 K-1
            (the default is to use the calibrated value).
        temp_bias : float, optional
            set to the value of the temperature bias you want to use
            (the default is to use the calibrated value).
        prcp_fac : float, optional
            set to the value of the precipitation factor you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration and
            the ones you are using at run time. If they don't match, it will
            raise an error. Set to "False" to suppress this check.
        params_filesuffix : str
            which calibrated parameters to read (default: '_with_ava')
        """

        super(MonthlyTIAvalancheModel, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.fl_id = fl_id  # which flowline are we the model of?
        self.gdir = gdir

        self.params_filesuffix = params_filesuffix

        if melt_f is None:
            melt_f = self.calib_params["melt_f"]

        if temp_bias is None:
            temp_bias = self.calib_params["temp_bias"]

        if prcp_fac is None:
            prcp_fac = self.calib_params["prcp_fac"]

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = self.calib_params["mb_global_params"]
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    msg = (
                        "You seem to use different mass balance parameters "
                        "than used for the calibration: "
                        f"you use cfg.PARAMS['{k}']={cfg.PARAMS[k]} while "
                        f"it was calibrated with cfg.PARAMS['{k}']={v}. "
                        "Set `check_calib_params=False` to ignore this "
                        "warning."
                    )
                    raise InvalidWorkflowError(msg)
            src = self.calib_params["baseline_climate_source"]
            src_calib = gdir.get_climate_info()["baseline_climate_source"]
            if src != src_calib:
                msg = (
                    f"You seem to have calibrated with the {src} "
                    f"climate data while this gdir was calibrated with "
                    f"{src_calib}. Set `check_calib_params=False` to "
                    f"ignore this warning."
                )
                raise InvalidWorkflowError(msg)

        self.melt_f = melt_f
        self.bias = bias

        # Global parameters
        self.t_solid = cfg.PARAMS["temp_all_solid"]
        self.t_liq = cfg.PARAMS["temp_all_liq"]
        self.t_melt = cfg.PARAMS["temp_melt"]

        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError("prcp_fac has to be above zero!")
        default_grad = cfg.PARAMS["temp_default_gradient"]

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        # Add avalanche data
        binned_data_file = gdir.get_filepath(
            "elevation_band_flowline", filesuffix="_fixed_dx"
        )
        binned_data = pd.read_csv(binned_data_file, index_col=0)
        self.ava_prpc_fac = binned_data.snowslide_1m.values

        # Private attrs
        # to allow prcp_fac to be changed after instantiation
        # prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = temp_bias

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with utils.ncDataset(fpath, mode="r") as nc:
            # time
            time = nc.variables["time"]
            time = cftime.num2date(time[:], time.units, calendar=time.calendar)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError("Climate data should be N full years")

            # We check for calendar years
            if (time[0].month != 1) or (time[-1].month != 12):
                raise InvalidWorkflowError(
                    "We now work exclusively with " "calendar years."
                )

            # Quick trick because we know the size of our array
            years = np.repeat(np.arange(time[-1].year - ny + 1, time[-1].year + 1), 12)
            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = years >= ys
            if ye is not None:
                try:
                    pok = pok & (years <= ye)
                except TypeError:
                    pok = years <= ye

            self.years = years[pok]
            self.months = np.tile(np.arange(1, 13), ny)[pok]

            # Read timeseries and correct it
            self.temp = nc.variables["temp"][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables["prcp"][pok].astype(np.float64) * self._prcp_fac

            grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.climate_source = nc.climate_source
            self.ys = self.years[0]
            self.ye = self.years[-1]

    def __repr__(self):
        """String Representation of the mass balance model"""
        summary = ["<oggm.MassBalanceModel>"]
        summary += ["  Class: " + self.__class__.__name__]
        summary += ["  Attributes:"]
        # Add all scalar attributes
        done = []
        for k in [
            "hemisphere",
            "climate_source",
            "melt_f",
            "prcp_fac",
            "temp_bias",
            "bias",
        ]:
            done.append(k)
            v = self.__getattribute__(k)
            if k == "climate_source":
                if v.endswith(".nc"):
                    v = os.path.basename(v)
            nofloat = ["hemisphere", "climate_source"]
            nbform = "    - {}: {}" if k in nofloat else "    - {}: {:.02f}"
            summary += [nbform.format(k, v)]
        for k, v in self.__dict__.items():
            if np.isscalar(v) and not k.startswith("_") and k not in done:
                nbform = "    - {}: {}"
                summary += [nbform.format(k, v)]
        return "\n".join(summary) + "\n"

    @property
    def monthly_melt_f(self):
        return self.melt_f * 365 / 12

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: cfg.PARAMS['prcp_fac'])

        Called factor to make clear that it is a multiplicative factor in
        contrast to the additive temperature bias
        """
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        # just to check that no invalid prcp_factors are used
        if np.any(np.asarray(new_prcp_fac) <= 0):
            raise InvalidParamsError("prcp_fac has to be above zero!")

        if len(np.atleast_1d(new_prcp_fac)) == 12:
            # OK so that's monthly stuff
            new_prcp_fac = np.tile(new_prcp_fac, len(self.prcp) // 12)

        self.prcp *= new_prcp_fac / self._prcp_fac
        self._prcp_fac = new_prcp_fac

    @property
    def temp_bias(self):
        """Add a temperature bias to the time series"""
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):
        if len(np.atleast_1d(new_temp_bias)) == 12:
            # OK so that's monthly stuff
            new_temp_bias = np.tile(new_temp_bias, len(self.temp) // 12)

        self.temp += new_temp_bias - self._temp_bias
        self._temp_bias = new_temp_bias

    @utils.lazy_property
    def calib_params(self):
        return self.gdir.read_json("mb_calib", filesuffix=self.params_filesuffix)

    def is_year_valid(self, year):
        return self.ys <= year <= self.ye

    def get_monthly_climate(self, heights, year=None):
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = utils.floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(y):
            raise ValueError(
                "year {} out of the valid time bounds: "
                "[{}, {}]".format(y, self.ys, self.ye)
            )
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        utils.clip_min(tempformelt, 0, out=tempformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * utils.clip_array(fac, 0, 1)

        # add precipitation correction factor from avalanching
        prcpsol = prcpsol * self.snowslide_1m.values

        return temp, tempformelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(year):
            raise ValueError(
                "year {} out of the valid time bounds: "
                "[{}, {}]".format(year, self.ys, self.ye)
            )
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError("Year {} not in record".format(int(year)))

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= heights.repeat(12).reshape(grad_temp.shape) - self.ref_hgt
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.t_melt
        utils.clip_min(temp2dformelt, 0, out=temp2dformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * utils.clip_array(fac, 0, 1)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (
            t.mean(axis=1),
            tmelt.sum(axis=1),
            prcp.sum(axis=1),
            prcpsol.sum(axis=1),
        )

    def get_monthly_mb(self, heights, year=None, add_climate=False, **kwargs):
        t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
        mb_month = prcpsol - self.monthly_melt_f * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        if add_climate:
            return (mb_month / SEC_IN_MONTH / self.rho, t, tmelt, prcp, prcpsol)
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        ava_fac = ava_prpc_fac.reshape(prcpsol.shape[0], 1)
        prcpsol *= ava_fac

        # calculate the positive & negative contributions from avalanches
        mask_add = ava_fac>1
        mask_rm = ava_fac<1
        ava_added = np.where(mask_add, prcpsol * (1 - 1/ava_fac), 0)
        ava_removed = np.where(mask_rm, prcpsol * (1 - 1/ava_fac), 0)

        mb_annual = np.sum(prcpsol - self.monthly_melt_f * tmelt, axis=1)
        mb_annual = (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
        if add_climate:
            return (
                mb_annual,
                t.mean(axis=1),
                tmelt.sum(axis=1),
                prcp.sum(axis=1),
                prcpsol.sum(axis=1),
                ava_added.sum(axis=1),
                ava_removed.sum(axis=1)
            )
        return mb_annual

    def get_annual_acc(self, heights, year=None, add_climate=False, **kwargs):
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)

        acc_annual = np.sum(prcpsol, axis=1)  / SEC_IN_YEAR / self.rho
        return acc_annual


@utils.entity_task(log, writes=["mb_calib"])
def mb_calibration_from_geodetic_mb_with_avalanches(
    gdir,
    *,
    ref_period=None,
    write_to_gdir=True,
    overwrite_gdir=True,
    override_missing=False,
    informed_threestep=False,
    mb_model_class=MonthlyTIAvalancheModel,
):
    """Calibrate for geodetic MB data from Hugonnet et al., 2021.

    For avalanches: we assume that the parameters were calibrated
    without avalanches prior to this call.

    The data table can be obtained with utils.get_geodetic_mb_dataframe().
    It is equivalent to the original data from Hugonnet, but has some outlier
    values filtered. See `this notebook` for more details.

    The problem of calibrating many unknown parameters on geodetic data is
    currently unsolved. This is OGGM's current take, based on trial and
    error and based on ideas from the literature.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to calibrate
    ref_period : str, default: PARAMS['geodetic_mb_period']
        one of '2000-01-01_2010-01-01', '2010-01-01_2020-01-01',
        '2000-01-01_2020-01-01'. If `ref_mb` is set, this should still match
        the same format but can be any date.
    write_to_gdir : bool
        whether to write the results of the calibration to the glacier
        directory. If True (the default), this will be saved as `mb_calib.json`
        and be used by the MassBalanceModel class as parameters in subsequent
        tasks.
    overwrite_gdir : bool
        if a `mb_calib.json` exists, this task won't overwrite it per default.
        Set this to True to enforce overwriting (i.e. with consequences for the
        future workflow).
    override_missing : scalar
        if the reference geodetic data is not available, use this value instead
        (mostly for testing with exotic datasets, but could be used to open
        the door to using other datasets).
    informed_threestep : bool
        the magic method Fabi found out one day before release.
        Overrides the calibrate_param order below.
    mb_model_class : MassBalanceModel class
        the MassBalanceModel to use for the calibration. Needs to use the
        same parameters as MonthlyTIModel (the default): melt_f,
        temp_bias, prcp_fac.

    Returns
    -------
    the calibrated parameters as dict
    """
    if not ref_period:
        ref_period = cfg.PARAMS["geodetic_mb_period"]

    # Get the reference data
    ref_mb_err = np.NaN
    try:
        ref_mb_df = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        ref_mb_df = ref_mb_df.loc[ref_mb_df["period"] == ref_period]
        # dmdtda: in meters water-equivalent per year -> we convert to kg m-2 yr-1
        ref_mb = ref_mb_df["dmdtda"].iloc[0] * 1000
        ref_mb_err = ref_mb_df["err_dmdtda"].iloc[0] * 1000
    except KeyError:
        if override_missing is None:
            raise
        ref_mb = override_missing

    if informed_threestep:
        temp_bias = 0
        if cfg.PARAMS['use_temp_bias_from_file']:
            climinfo = gdir.get_climate_info()
            if 'w5e5' not in climinfo['baseline_climate_source'].lower():
                raise InvalidWorkflowError('use_temp_bias_from_file currently '
                                        'only available for W5E5 data.')
        bias_df = get_temp_bias_dataframe()
        ref_lon = climinfo['baseline_climate_ref_pix_lon']
        ref_lat = climinfo['baseline_climate_ref_pix_lat']
        # Take nearest
        dis = ((bias_df.lon_val - ref_lon)**2 + (bias_df.lat_val - ref_lat)**2)**0.5
        sel_df = bias_df.iloc[np.argmin(dis)]
        temp_bias = sel_df['median_temp_bias_w_err_grouped']
        assert np.isfinite(temp_bias), 'Temp bias not finite?'

        if not cfg.PARAMS['use_temp_bias_from_file']:
            raise InvalidParamsError('With `informed_threestep` you need to '
                                     'set `use_temp_bias_from_file`.')
        if not cfg.PARAMS['use_winter_prcp_fac']:
            raise InvalidParamsError('With `informed_threestep` you need to '
                                     'set `use_winter_prcp_fac`.')

        # Some magic heuristics - we just decide to calibrate
        # precip -> melt_f -> temp but informed by previous data.

        # Temp bias was decided anyway, we keep as previous value and
        # allow it to vary as last resort

        # We use the precip factor but allow it to vary between 0.8, 1.2 of
        # the previous value (uncertainty).
        prcp_fac = decide_winter_precip_factor(gdir)
        mi, ma = cfg.PARAMS['prcp_fac_min'], cfg.PARAMS['prcp_fac_max']

        # these were the 'arbitrary' bounds imposed in Schuster et al., 2023. 
        prcp_fac_min = clip_scalar(prcp_fac * 0.8, mi, ma)
        prcp_fac_max = clip_scalar(prcp_fac * 1.2, mi, ma)

        # If we want to allow them to vary more (but keeping the minimum and maximum values)
        #prcp_fac_min = mi
        #prcp_fac_max = ma

        return mb_calibration_from_scalar_mb(gdir,
                                             ref_mb=ref_mb,
                                             ref_mb_err=ref_mb_err,
                                             ref_period=ref_period,
                                             write_to_gdir=write_to_gdir,
                                             overwrite_gdir=overwrite_gdir,
                                             calibrate_param1='prcp_fac',
                                             calibrate_param2='melt_f',
                                             calibrate_param3='temp_bias',
                                             prcp_fac=prcp_fac,
                                             prcp_fac_min=prcp_fac_min,
                                             prcp_fac_max=prcp_fac_max,
                                             temp_bias=temp_bias,
                                             mb_model_class=mb_model_class,
                                             filesuffix='_with_ava',
                                             )
    
    else: 
        # We assume it has been calibrated
        calib_params = gdir.read_json("mb_calib")

        return mb_calibration_from_scalar_mb(
            gdir,
            ref_mb=ref_mb,
            ref_mb_err=ref_mb_err,
            ref_period=ref_period,
            write_to_gdir=write_to_gdir,
            overwrite_gdir=overwrite_gdir,
            calibrate_param1="prcp_fac",
            calibrate_param2="melt_f",
            calibrate_param3="temp_bias",
            prcp_fac=calib_params["prcp_fac"],
            melt_f=calib_params["melt_f"],
            temp_bias=calib_params["temp_bias"],
            mb_model_class=mb_model_class,
            filesuffix='_with_ava',
        )


class MonthlyTIAvalancheModel_2D(MassBalanceModel):
    """Monthly temperature index model."""

    def __init__(
        self,
        gdir,
        filename="climate_historical",
        input_filesuffix="",
        fl_id=None,
        melt_f=None,
        temp_bias=None,
        prcp_fac=None,
        bias=0,
        ys=None,
        ye=None,
        repeat=False,
        check_calib_params=True,
        params_filesuffix='_with_ava_2d',
    ):
        """Initialize.

        Parameters
        ----------
        gdir : GlacierDirectory
            the glacier directory
        filename : str, optional
            set to a different BASENAME if you want to use alternative climate
            data. Default is 'climate_historical'
        input_filesuffix : str, optional
            append a suffix to the filename (useful for GCM runs).
        fl_id : int, optional
            if this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            set to the value of the melt factor you want to use,
            here the unit is kg m-2 day-1 K-1
            (the default is to use the calibrated value).
        temp_bias : float, optional
            set to the value of the temperature bias you want to use
            (the default is to use the calibrated value).
        prcp_fac : float, optional
            set to the value of the precipitation factor you want to use
            (the default is to use the calibrated value).
        bias : float, optional
            set to the alternative value of the calibration bias [mm we yr-1]
            you want to use (the default is to use the calibrated value)
            Note that this bias is *substracted* from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        ys : int
            The start of the climate period where the MB model is valid
            (default: the period with available data)
        ye : int
            The end of the climate period where the MB model is valid
            (default: the period with available data)
        repeat : bool
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration and
            the ones you are using at run time. If they don't match, it will
            raise an error. Set to "False" to suppress this check.
        params_filesuffix : str
            which calibrated parameters to read (default: '_with_ava')
        """

        super(MonthlyTIAvalancheModel_2D, self).__init__()
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.fl_id = fl_id  # which flowline are we the model of?
        self.gdir = gdir

        self.params_filesuffix = params_filesuffix

        if melt_f is None:
            melt_f = self.calib_params["melt_f"]

        if temp_bias is None:
            temp_bias = self.calib_params["temp_bias"]

        if prcp_fac is None:
            prcp_fac = self.calib_params["prcp_fac"]

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = self.calib_params["mb_global_params"]
            for k, v in mb_calib.items():
                if v != cfg.PARAMS[k]:
                    msg = (
                        "You seem to use different mass balance parameters "
                        "than used for the calibration: "
                        f"you use cfg.PARAMS['{k}']={cfg.PARAMS[k]} while "
                        f"it was calibrated with cfg.PARAMS['{k}']={v}. "
                        "Set `check_calib_params=False` to ignore this "
                        "warning."
                    )
                    raise InvalidWorkflowError(msg)
            src = self.calib_params["baseline_climate_source"]
            src_calib = gdir.get_climate_info()["baseline_climate_source"]
            if src != src_calib:
                msg = (
                    f"You seem to have calibrated with the {src} "
                    f"climate data while this gdir was calibrated with "
                    f"{src_calib}. Set `check_calib_params=False` to "
                    f"ignore this warning."
                )
                raise InvalidWorkflowError(msg)

        self.melt_f = melt_f
        self.bias = bias

        # Global parameters
        self.t_solid = cfg.PARAMS["temp_all_solid"]
        self.t_liq = cfg.PARAMS["temp_all_liq"]
        self.t_melt = cfg.PARAMS["temp_melt"]

        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError("prcp_fac has to be above zero!")
        default_grad = cfg.PARAMS["temp_default_gradient"]

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        # Add avalanche data
        gridded_data_file = gdir.get_filepath("gridded_data")
        with xr.open_dataset(gridded_data_file) as gridded_data:
            gridded_data = gridded_data.load()

        self.ava_prpc_fac = gridded_data.snowslide_1m.values

        # Private attrs
        # to allow prcp_fac to be changed after instantiation
        # prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = temp_bias

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with utils.ncDataset(fpath, mode="r") as nc:
            # time
            time = nc.variables["time"]
            time = cftime.num2date(time[:], time.units, calendar=time.calendar)
            ny, r = divmod(len(time), 12)
            if r != 0:
                raise ValueError("Climate data should be N full years")

            # We check for calendar years
            if (time[0].month != 1) or (time[-1].month != 12):
                raise InvalidWorkflowError(
                    "We now work exclusively with " "calendar years."
                )

            # Quick trick because we know the size of our array
            years = np.repeat(np.arange(time[-1].year - ny + 1, time[-1].year + 1), 12)
            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = years >= ys
            if ye is not None:
                try:
                    pok = pok & (years <= ye)
                except TypeError:
                    pok = years <= ye

            self.years = years[pok]
            self.months = np.tile(np.arange(1, 13), ny)[pok]

            # Read timeseries and correct it
            self.temp = nc.variables["temp"][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables["prcp"][pok].astype(np.float64) * self._prcp_fac

            grad = self.prcp * 0 + default_grad
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.climate_source = nc.climate_source
            self.ys = self.years[0]
            self.ye = self.years[-1]

    def __repr__(self):
        """String Representation of the mass balance model"""
        summary = ["<oggm.MassBalanceModel>"]
        summary += ["  Class: " + self.__class__.__name__]
        summary += ["  Attributes:"]
        # Add all scalar attributes
        done = []
        for k in [
            "hemisphere",
            "climate_source",
            "melt_f",
            "prcp_fac",
            "temp_bias",
            "bias",
        ]:
            done.append(k)
            v = self.__getattribute__(k)
            if k == "climate_source":
                if v.endswith(".nc"):
                    v = os.path.basename(v)
            nofloat = ["hemisphere", "climate_source"]
            nbform = "    - {}: {}" if k in nofloat else "    - {}: {:.02f}"
            summary += [nbform.format(k, v)]
        for k, v in self.__dict__.items():
            if np.isscalar(v) and not k.startswith("_") and k not in done:
                nbform = "    - {}: {}"
                summary += [nbform.format(k, v)]
        return "\n".join(summary) + "\n"

    @property
    def monthly_melt_f(self):
        return self.melt_f * 365 / 12

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: cfg.PARAMS['prcp_fac'])

        Called factor to make clear that it is a multiplicative factor in
        contrast to the additive temperature bias
        """
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        # just to check that no invalid prcp_factors are used
        if np.any(np.asarray(new_prcp_fac) <= 0):
            raise InvalidParamsError("prcp_fac has to be above zero!")

        if len(np.atleast_1d(new_prcp_fac)) == 12:
            # OK so that's monthly stuff
            new_prcp_fac = np.tile(new_prcp_fac, len(self.prcp) // 12)

        self.prcp *= new_prcp_fac / self._prcp_fac
        self._prcp_fac = new_prcp_fac

    @property
    def temp_bias(self):
        """Add a temperature bias to the time series"""
        return self._temp_bias

    @temp_bias.setter
    def temp_bias(self, new_temp_bias):
        if len(np.atleast_1d(new_temp_bias)) == 12:
            # OK so that's monthly stuff
            new_temp_bias = np.tile(new_temp_bias, len(self.temp) // 12)

        self.temp += new_temp_bias - self._temp_bias
        self._temp_bias = new_temp_bias

    @utils.lazy_property
    def calib_params(self):
        return self.gdir.read_json("mb_calib", filesuffix=self.params_filesuffix)

    def is_year_valid(self, year):
        return self.ys <= year <= self.ye

    def get_monthly_climate(self, heights, year=None):
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """

        y, m = utils.floatyear_to_date(year)
        if self.repeat:
            y = self.ys + (y - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(y):
            raise ValueError(
                "year {} out of the valid time bounds: "
                "[{}, {}]".format(y, self.ys, self.ye)
            )
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        npix = len(heights)
        temp = np.ones(npix) * itemp + igrad * (heights - self.ref_hgt)
        tempformelt = temp - self.t_melt
        utils.clip_min(tempformelt, 0, out=tempformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.ones(npix) * iprcp
        fac = 1 - (temp - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * utils.clip_array(fac, 0, 1)

        # add precipitation correction factor from avalanching
        prcpsol = prcpsol * self.snowslide_1m.values

        return temp, tempformelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(year):
            raise ValueError(
                "year {} out of the valid time bounds: "
                "[{}, {}]".format(year, self.ys, self.ye)
            )
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError("Year {} not in record".format(int(year)))

        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        # Compute temp and tempformelt (temperature above melting threshold)
        heights = np.asarray(heights)
        npix = len(heights)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= heights.repeat(12).reshape(grad_temp.shape) - self.ref_hgt
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = temp2d - self.t_melt
        utils.clip_min(temp2dformelt, 0, out=temp2dformelt)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        fac = 1 - (temp2d - self.t_solid) / (self.t_liq - self.t_solid)
        prcpsol = prcp * utils.clip_array(fac, 0, 1)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        (temp, tempformelt, prcp, prcpsol)
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)
        return (
            t.mean(axis=1),
            tmelt.sum(axis=1),
            prcp.sum(axis=1),
            prcpsol.sum(axis=1),
        )

    def get_monthly_mb(self, heights, year=None, add_climate=False, **kwargs):
        t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)
        mb_month = prcpsol - self.monthly_melt_f * tmelt
        mb_month -= self.bias * SEC_IN_MONTH / SEC_IN_YEAR
        if add_climate:
            return (mb_month / SEC_IN_MONTH / self.rho, t, tmelt, prcp, prcpsol)
        return mb_month / SEC_IN_MONTH / self.rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        ava_fac = ava_prpc_fac.reshape(prcpsol.shape[0], 1)
        prcpsol *= ava_fac

        # calculate the positive & negative contributions from avalanches
        mask_add = ava_fac>1
        mask_rm = ava_fac<1
        ava_added = np.where(mask_add, prcpsol * (ava_fac - 1), 0)
        ava_removed = np.where(mask_rm, prcpsol * (ava_fac - 1), 0)

        mb_annual = np.sum(prcpsol - self.monthly_melt_f * tmelt, axis=1)
        mb_annual = (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
        if add_climate:
            return (
                mb_annual,
                t.mean(axis=1),
                tmelt.sum(axis=1),
                prcp.sum(axis=1),
                prcpsol.sum(axis=1),
                ava_added.sum(axis=1),
                ava_removed.sum(axis=1)
            )
        return mb_annual

    def get_annual_acc(self, heights, year=None, add_climate=False, **kwargs):
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)

        acc_annual = np.sum(prcpsol, axis=1)  / SEC_IN_YEAR / self.rho
        return acc_annual


@utils.entity_task(log, writes=["mb_calib"])
def mb_calibration_from_geodetic_mb_with_avalanches_2D(
    gdir,
    *,
    ref_period=None,
    write_to_gdir=True,
    overwrite_gdir=True,
    override_missing=False,
    mb_model_class=MonthlyTIAvalancheModel_2D,
):
    """Calibrate for geodetic MB data from Hugonnet et al., 2021.

    For avalanches: we assume that the parameters were calibrated
    without avalanches prior to this call.

    The data table can be obtained with utils.get_geodetic_mb_dataframe().
    It is equivalent to the original data from Hugonnet, but has some outlier
    values filtered. See `this notebook` for more details.

    The problem of calibrating many unknown parameters on geodetic data is
    currently unsolved. This is OGGM's current take, based on trial and
    error and based on ideas from the literature.

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to calibrate
    ref_period : str, default: PARAMS['geodetic_mb_period']
        one of '2000-01-01_2010-01-01', '2010-01-01_2020-01-01',
        '2000-01-01_2020-01-01'. If `ref_mb` is set, this should still match
        the same format but can be any date.
    write_to_gdir : bool
        whether to write the results of the calibration to the glacier
        directory. If True (the default), this will be saved as `mb_calib.json`
        and be used by the MassBalanceModel class as parameters in subsequent
        tasks.
    overwrite_gdir : bool
        if a `mb_calib.json` exists, this task won't overwrite it per default.
        Set this to True to enforce overwriting (i.e. with consequences for the
        future workflow).
    override_missing : scalar
        if the reference geodetic data is not available, use this value instead
        (mostly for testing with exotic datasets, but could be used to open
        the door to using other datasets).
    mb_model_class : MassBalanceModel class
        the MassBalanceModel to use for the calibration. Needs to use the
        same parameters as MonthlyTIModel (the default): melt_f,
        temp_bias, prcp_fac.

    Returns
    -------
    the calibrated parameters as dict
    """
    if not ref_period:
        ref_period = cfg.PARAMS["geodetic_mb_period"]

    # Get the reference data
    ref_mb_err = np.NaN
    try:
        ref_mb_df = utils.get_geodetic_mb_dataframe().loc[gdir.rgi_id]
        ref_mb_df = ref_mb_df.loc[ref_mb_df["period"] == ref_period]
        # dmdtda: in meters water-equivalent per year -> we convert to kg m-2 yr-1
        ref_mb = ref_mb_df["dmdtda"].iloc[0] * 1000
        ref_mb_err = ref_mb_df["err_dmdtda"].iloc[0] * 1000
    except KeyError:
        if override_missing is None:
            raise
        ref_mb = override_missing

    # We assume it has been calibrated
    calib_params = gdir.read_json("mb_calib")

    return mb_calibration_from_scalar_mb(
        gdir,
        ref_mb=ref_mb,
        ref_mb_err=ref_mb_err,
        ref_period=ref_period,
        write_to_gdir=write_to_gdir,
        overwrite_gdir=overwrite_gdir,
        calibrate_param1="prcp_fac",
        calibrate_param2="melt_f",
        calibrate_param3="temp_bias",
        prcp_fac=calib_params["prcp_fac"],
        melt_f=calib_params["melt_f"],
        temp_bias=calib_params["temp_bias"],
        mb_model_class=mb_model_class,
        filesuffix='_with_ava',
    )


@utils.global_task(log)
@utils.compile_to_netcdf(log)
def compile_run_output(gdirs, path=True, input_filesuffix='',
                       use_compression=True):
    """Compiles the output of the model runs of several gdirs into one file.

    Updated from utils._workflow to include mean_snowslide output

    Parameters
    ----------
    gdirs : list of :py:class:`oggm.GlacierDirectory` objects
        the glacier directories to process
    path : str
        where to store (default is on the working dir).
        Set to `False` to disable disk storage.
    input_filesuffix : str
        the filesuffix of the files to be compiled
    use_compression : bool
        use zlib compression on the output netCDF files

    Returns
    -------
    ds : :py:class:`xarray.Dataset`
        compiled output
    """

    # Get the dimensions of all this
    rgi_ids = [gd.rgi_id for gd in gdirs]

    # To find the longest time, we have to open all files unfortunately, we
    # also create a list of all data variables (in case not all files contain
    # the same data variables), and finally we decide on the name of "3d"
    # variables in case we have daily
    time_info = {}
    time_keys = ['hydro_year', 'hydro_month', 'calendar_year', 'calendar_month']
    allowed_data_vars = ['volume_m3', 'volume_bsl_m3', 'volume_bwl_m3',
                         'volume_m3_min_h',  # only here for back compatibility
                         # as it is a variable in gdirs v1.6 2023.1
                         'area_m2', 'area_m2_min_h', 'length_m', 'calving_m3',
                         'calving_rate_myr', 'off_area',
                         'on_area', 'model_mb', 'is_fixed_geometry_spinup', 'mean_snowslide']
    for gi in range(10):
        allowed_data_vars += [f'terminus_thick_{gi}']
    # this hydro variables can be _monthly or _daily
    hydro_vars = ['melt_off_glacier', 'melt_on_glacier',
                  'liq_prcp_off_glacier', 'liq_prcp_on_glacier',
                  'snowfall_off_glacier', 'snowfall_on_glacier',
                  'avalanche_added', 'avalanche_removed',
                  'melt_residual_off_glacier', 'melt_residual_on_glacier',
                  'snow_bucket', 'residual_mb']
    for v in hydro_vars:
        allowed_data_vars += [v]
        allowed_data_vars += [v + '_monthly']
        allowed_data_vars += [v + '_daily']
    data_vars = {}
    name_2d_dim = 'month_2d'
    contains_3d_data = False
    for gd in gdirs:
        fp = gd.get_filepath('model_diagnostics', filesuffix=input_filesuffix)
        try:
            with utils.ncDataset(fp) as ds:
                time = ds.variables['time'][:]
                if 'time' not in time_info:
                    time_info['time'] = time
                    for cn in time_keys:
                        time_info[cn] = ds.variables[cn][:]
                else:
                    # Here we may need to append or add stuff
                    ot = time_info['time']
                    if time[0] > ot[-1] or ot[-1] < time[0]:
                        raise InvalidWorkflowError('Trying to compile output '
                                                   'without overlap.')
                    if time[-1] > ot[-1]:
                        p = np.nonzero(time == ot[-1])[0][0] + 1
                        time_info['time'] = np.append(ot, time[p:])
                        for cn in time_keys:
                            time_info[cn] = np.append(time_info[cn],
                                                      ds.variables[cn][p:])
                    if time[0] < ot[0]:
                        p = np.nonzero(time == ot[0])[0][0]
                        time_info['time'] = np.append(time[:p], ot)
                        for cn in time_keys:
                            time_info[cn] = np.append(ds.variables[cn][:p],
                                                      time_info[cn])

                # check if their are new data variables and add them
                for vn in ds.variables:
                    # exclude time variables
                    if vn in ['month_2d', 'calendar_month_2d',
                              'hydro_month_2d']:
                        name_2d_dim = 'month_2d'
                        contains_3d_data = True
                    elif vn in ['day_2d', 'calendar_day_2d', 'hydro_day_2d']:
                        name_2d_dim = 'day_2d'
                        contains_3d_data = True
                    elif vn in allowed_data_vars:
                        # check if data variable is new
                        if vn not in data_vars.keys():
                            data_vars[vn] = dict()
                            data_vars[vn]['dims'] = ds.variables[vn].dimensions
                            data_vars[vn]['attrs'] = dict()
                            for attr in ds.variables[vn].ncattrs():
                                if attr not in ['_FillValue', 'coordinates',
                                                'dtype']:
                                    data_vars[vn]['attrs'][attr] = getattr(
                                        ds.variables[vn], attr)
                    elif vn not in ['time'] + time_keys:
                        # This check has future developments in mind.
                        # If you end here it means the current data variable is
                        # not under the allowed_data_vars OR not under the
                        # defined time dimensions. If it is a new data variable
                        # add it to allowed_data_vars above (also add it to
                        # test_compile_run_output). If it is a new dimension
                        # handle it in the if/elif statements.
                        raise InvalidParamsError(f'The data variable "{vn}" '
                                                 'is not known. Is it new or '
                                                 'is it a new dimension? '
                                                 'Check comment above this '
                                                 'raise for more info!')

            # If this worked, keep it as template
            ppath = fp
        except FileNotFoundError:
            pass

    if 'time' not in time_info:
        raise RuntimeError('Found no valid glaciers!')

    # OK found it, open it and prepare the output
    with xr.open_dataset(ppath) as ds_diag:

        # Prepare output
        ds = xr.Dataset()

        # Global attributes
        ds.attrs['description'] = 'OGGM model output'
        ds.attrs['oggm_version'] = __version__
        ds.attrs['calendar'] = '365-day no leap'
        ds.attrs['creation_date'] = strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # Copy coordinates
        time = time_info['time']
        ds.coords['time'] = ('time', time)
        ds['time'].attrs['description'] = 'Floating year'
        # New coord
        ds.coords['rgi_id'] = ('rgi_id', rgi_ids)
        ds['rgi_id'].attrs['description'] = 'RGI glacier identifier'
        # This is just taken from there
        for cn in ['hydro_year', 'hydro_month',
                   'calendar_year', 'calendar_month']:
            ds.coords[cn] = ('time', time_info[cn])
            ds[cn].attrs['description'] = ds_diag[cn].attrs['description']

        # Prepare the 2D variables
        shape = (len(time), len(rgi_ids))
        out_2d = dict()
        for vn in data_vars:
            if name_2d_dim in data_vars[vn]['dims']:
                continue
            var = dict()
            var['data'] = np.full(shape, np.nan)
            var['attrs'] = data_vars[vn]['attrs']
            out_2d[vn] = var

        # 1D Variables
        out_1d = dict()
        for vn, attrs in [('water_level', {'description': 'Calving water level',
                                           'units': 'm'}),
                          ('glen_a', {'description': 'Simulation Glen A',
                                      'units': ''}),
                          ('fs', {'description': 'Simulation sliding parameter',
                                  'units': ''}),
                          ]:
            var = dict()
            var['data'] = np.full(len(rgi_ids), np.nan)
            var['attrs'] = attrs
            out_1d[vn] = var

        # Maybe 3D?
        out_3d = dict()
        if contains_3d_data:
            # We have some 3d vars
            month_2d = ds_diag[name_2d_dim]
            ds.coords[name_2d_dim] = (name_2d_dim, month_2d.data)
            cn = f'calendar_{name_2d_dim}'
            ds.coords[cn] = (name_2d_dim, ds_diag[cn].values)

            shape = (len(time), len(month_2d), len(rgi_ids))
            for vn in data_vars:
                if name_2d_dim not in data_vars[vn]['dims']:
                    continue
                var = dict()
                var['data'] = np.full(shape, np.nan)
                var['attrs'] = data_vars[vn]['attrs']
                out_3d[vn] = var

    # Read out
    for i, gdir in enumerate(gdirs):
        try:
            ppath = gdir.get_filepath('model_diagnostics',
                                      filesuffix=input_filesuffix)
            with utils.ncDataset(ppath) as ds_diag:
                it = ds_diag.variables['time'][:]
                a = np.nonzero(time == it[0])[0][0]
                b = np.nonzero(time == it[-1])[0][0] + 1
                for vn, var in out_2d.items():
                    # try statement if some data variables not in all files
                    try:
                        var['data'][a:b, i] = ds_diag.variables[vn][:]
                    except KeyError:
                        pass
                for vn, var in out_3d.items():
                    # try statement if some data variables not in all files
                    try:
                        var['data'][a:b, :, i] = ds_diag.variables[vn][:]
                    except KeyError:
                        pass
                for vn, var in out_1d.items():
                    var['data'][i] = ds_diag.getncattr(vn)
        except FileNotFoundError:
            pass

    # To xarray
    for vn, var in out_2d.items():
        # Backwards compatibility - to remove one day...
        for r in ['_m3', '_m2', '_myr', '_m']:
            # Order matters
            vn = regexp.sub(r + '$', '', vn)
        ds[vn] = (('time', 'rgi_id'), var['data'])
        ds[vn].attrs = var['attrs']
    for vn, var in out_3d.items():
        ds[vn] = (('time', name_2d_dim, 'rgi_id'), var['data'])
        ds[vn].attrs = var['attrs']
    for vn, var in out_1d.items():
        ds[vn] = (('rgi_id', ), var['data'])
        ds[vn].attrs = var['attrs']

    # To file?
    if path:
        enc_var = {'dtype': 'float32'}
        if use_compression:
            enc_var['complevel'] = 5
            enc_var['zlib'] = True
        encoding = {v: enc_var for v in ds.data_vars}
        ds.to_netcdf(path, encoding=encoding)

    return ds



@utils.entity_task(log)
def run_with_hydro_ava(gdir, run_task=None, store_monthly_hydro=False,
                   fixed_geometry_spinup_yr=None, ref_area_from_y0=False,
                   ref_area_yr=None, ref_geometry_filesuffix=None,
                   **kwargs):
    """Run the flowline model and add hydro diagnostics.

    - Added output variables - avalanche_added & avalanche_removed

    Parameters
    ----------
    run_task : func
        any of the `run_*`` tasks in the oggm.flowline module.
        The mass balance model used needs to have the `add_climate` output
        kwarg available though.
    store_monthly_hydro : bool
        also compute monthly hydrological diagnostics. The monthly outputs
        are stored in 2D fields (years, months)
    ref_area_yr : int
        the hydrological output is computed over a reference area, which
        per default is the largest area covered by the glacier in the simulation
        period. Use this kwarg to force a specific area to the state of the
        glacier at the provided simulation year.
    ref_area_from_y0 : bool
        overwrite ref_area_yr to the first year of the timeseries
    ref_geometry_filesuffix : str
        this kwarg allows to copy the reference area from a previous simulation
        (useful for projections with historical spinup for example).
        Set to a model_geometry file filesuffix that is present in the
        current directory (e.g. `_historical` for pre-processed gdirs).
        If set, ref_area_yr and ref_area_from_y0 refer to this file instead.
    fixed_geometry_spinup_yr : int
        if set to an integer, the model will artificially prolongate
        all outputs of run_until_and_store to encompass all time stamps
        starting from the chosen year. The only output affected are the
        glacier wide diagnostic files - all other outputs are set
        to constants during "spinup"
    **kwargs : all valid kwargs for ``run_task``
    """

    # Make sure it'll return something
    kwargs['return_value'] = True

    # Check that kwargs and params are compatible
    if kwargs.get('store_monthly_step', False):
        raise InvalidParamsError('run_with_hydro only compatible with '
                                 'store_monthly_step=False.')
    if kwargs.get('mb_elev_feedback', 'annual') != 'annual':
        raise InvalidParamsError('run_with_hydro only compatible with '
                                 "mb_elev_feedback='annual' (yes, even "
                                 "when asked for monthly hydro output).")
    if not cfg.PARAMS['store_model_geometry']:
        raise InvalidParamsError('run_with_hydro only works with '
                                 "PARAMS['store_model_geometry'] = True "
                                 "for now.")

    if fixed_geometry_spinup_yr is not None:
        kwargs['fixed_geometry_spinup_yr'] = fixed_geometry_spinup_yr

    out = run_task(gdir, **kwargs) # ye=2020, fixed_geometry_spinup_yr=2000, climate_filename='climate_historical', climate_input_filesuffix='', output_filesuffix=f'_ava_2000-2020', mb_model_class=MonthlyTIAvalancheModel

    if out is None:
        raise InvalidWorkflowError('The run task ({}) did not run '
                                   'successfully.'.format(run_task.__name__))

    do_spinup = fixed_geometry_spinup_yr is not None
    if do_spinup:
        start_dyna_model_yr = out.y0

    # Mass balance model used during the run
    mb_mod = out.mb_model

    # Glacier geometry during the run
    suffix = kwargs.get('output_filesuffix', '')

    # We start by fetching the reference model geometry
    # The one we just computed
    fmod = FileModel(gdir.get_filepath('model_geometry', filesuffix=suffix))
    # The last one is the final state - we can't compute MB for that
    years = fmod.years[:-1]

    if ref_geometry_filesuffix:
        if not ref_area_from_y0 and ref_area_yr is None:
            raise InvalidParamsError('If `ref_geometry_filesuffix` is set, '
                                     'users need to specify `ref_area_from_y0`'
                                     ' or `ref_area_yr`')
        # User provided
        fmod_ref = FileModel(gdir.get_filepath('model_geometry',
                                               filesuffix=ref_geometry_filesuffix))
    else:
        # ours as well
        fmod_ref = fmod

    # Check input
    if ref_area_from_y0:
        ref_area_yr = fmod_ref.years[0]

    # Geometry at year yr to start with + off-glacier snow bucket
    if ref_area_yr is not None:
        if ref_area_yr not in fmod_ref.years:
            raise InvalidParamsError('The chosen ref_area_yr is not '
                                     'available!')
        fmod_ref.run_until(ref_area_yr)

    bin_area_2ds = []
    bin_elev_2ds = []
    ref_areas = []
    snow_buckets = []
    for fl in fmod_ref.fls:
        # Glacier area on bins
        bin_area = fl.bin_area_m2
        ref_areas.append(bin_area)
        snow_buckets.append(bin_area * 0)

        # Output 2d data
        shape = len(years), len(bin_area)
        bin_area_2ds.append(np.empty(shape, np.float64))
        bin_elev_2ds.append(np.empty(shape, np.float64))

    # Ok now fetch all geometry data in a first loop
    # We do that because we might want to get the largest possible area (default)
    # and we want to minimize the number of calls to run_until
    for i, yr in enumerate(years):
        fmod.run_until(yr)
        for fl_id, (fl, bin_area_2d, bin_elev_2d) in \
                enumerate(zip(fmod.fls, bin_area_2ds, bin_elev_2ds)):
            # Time varying bins
            bin_area_2d[i, :] = fl.bin_area_m2
            bin_elev_2d[i, :] = fl.surface_h

    if ref_area_yr is None:
        # Ok we get the max area instead
        for ref_area, bin_area_2d in zip(ref_areas, bin_area_2ds):
            ref_area[:] = bin_area_2d.max(axis=0)

    # Ok now we have arrays, we can work with that
    # -> second time varying loop is for mass balance
    months = [1]
    seconds = cfg.SEC_IN_YEAR
    ntime = len(years) + 1
    oshape = (ntime, 1)
    if store_monthly_hydro:
        months = np.arange(1, 13)
        seconds = cfg.SEC_IN_MONTH
        oshape = (ntime, 12)

    out = {
        'off_area': {
            'description': 'Off-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'on_area': {
            'description': 'On-glacier area',
            'unit': 'm 2',
            'data': np.zeros(ntime),
        },
        'melt_off_glacier': {
            'description': 'Off-glacier melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_on_glacier': {
            'description': 'On-glacier melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_off_glacier': {
            'description': 'Off-glacier melt due to MB model residual',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'melt_residual_on_glacier': {
            'description': 'On-glacier melt due to MB model residual',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_off_glacier': {
            'description': 'Off-glacier liquid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'liq_prcp_on_glacier': {
            'description': 'On-glacier liquid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snowfall_off_glacier': {
            'description': 'Off-glacier solid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snowfall_on_glacier': {
            'description': 'On-glacier solid precipitation',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'avalanche_added': {
            'description': 'On-glacier snow added by avalanches',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'avalanche_removed': {
            'description': 'On-glacier snow removed by avalanches',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
        'snow_bucket': {
            'description': 'Off-glacier snow reservoir (state variable)',
            'unit': 'kg',
            'data': np.zeros(oshape),
        },
        'model_mb': {
            'description': 'Annual mass balance from dynamical model',
            'unit': 'kg yr-1',
            'data': np.zeros(ntime),
        },
        'residual_mb': {
            'description': 'Difference (before correction) between mb model and dyn model melt',
            'unit': 'kg yr-1',
            'data': np.zeros(oshape),
        },
    }

    # Initialize
    fmod.run_until(years[0])
    prev_model_vol = fmod.volume_m3

    for i, yr in enumerate(years):

        # Now the loop over the months
        for m in months:

            # A bit silly but avoid double counting in monthly ts
            off_area_out = 0
            on_area_out = 0

            for fl_id, (ref_area, snow_bucket, bin_area_2d, bin_elev_2d) in \
                    enumerate(zip(ref_areas, snow_buckets, bin_area_2ds, bin_elev_2ds)):

                bin_area = bin_area_2d[i, :]
                bin_elev = bin_elev_2d[i, :]

                # Make sure we have no negative contribution when glaciers are out
                off_area = utils.clip_min(ref_area - bin_area, 0)

                try:
                    if store_monthly_hydro:
                        flt_yr = utils.date_to_floatyear(int(yr), m)
                        mb_out = mb_mod.get_monthly_mb(bin_elev, fl_id=fl_id,
                                                       year=flt_yr,
                                                       add_climate=True)
                        mb, _, _, prcp, prcpsol = mb_out
                    else:
                        mb_out = mb_mod.get_annual_mb(bin_elev, fl_id=fl_id,
                                                      year=yr, add_climate=True)
                        mb, _, _, prcp, prcpsol, ava_added, ava_removed  = mb_out
                except ValueError as e:
                    if 'too many values to unpack' in str(e):
                        raise InvalidWorkflowError('Run with hydro needs a MB '
                                                   'model able to add climate '
                                                   'info to `get_annual_mb`.')
                    raise

                # Here we use mass (kg yr-1) not ice volume
                mb *= seconds * cfg.PARAMS['ice_density']

                # Bias of the mb model is a fake melt term that we need to deal with
                # This is here for correction purposes later
                mb_bias = mb_mod.bias * seconds / cfg.SEC_IN_YEAR

                liq_prcp_on_g = (prcp - prcpsol) * bin_area
                liq_prcp_off_g = (prcp - prcpsol) * off_area

                prcpsol_on_g = prcpsol * bin_area
                ava_added = ava_added * bin_area
                ava_removed = ava_removed * bin_area
                prcpsol_off_g = prcpsol * off_area

                # IMPORTANT: this does not guarantee that melt cannot be negative
                # the reason is the MB residual that here can only be understood
                # as a fake melt process.
                # In particular at the monthly scale this can lead to negative
                # or winter positive melt - we try to mitigate this
                # issue at the end of the year
                melt_on_g = (prcpsol - mb) * bin_area
                melt_off_g = (prcpsol - mb) * off_area

                if mb_mod.bias == 0:
                    # melt_on_g and melt_off_g can be negative, but the absolute
                    # values are very small. so we clip them to zero
                    melt_on_g = utils.clip_min(melt_on_g, 0)
                    melt_off_g = utils.clip_min(melt_off_g, 0)

                # This is the bad boy
                bias_on_g = mb_bias * bin_area
                bias_off_g = mb_bias * off_area

                # Update bucket with accumulation and melt
                snow_bucket += prcpsol_off_g
                # It can only melt that much
                melt_off_g = np.where((snow_bucket - melt_off_g) >= 0, melt_off_g, snow_bucket)
                # Update bucket
                snow_bucket -= melt_off_g

                # This is recomputed each month but well
                off_area_out += np.sum(off_area)
                on_area_out += np.sum(bin_area)

                # Monthly out
                out['melt_off_glacier']['data'][i, m-1] += np.sum(melt_off_g)
                out['melt_on_glacier']['data'][i, m-1] += np.sum(melt_on_g)
                out['melt_residual_off_glacier']['data'][i, m-1] += np.sum(bias_off_g)
                out['melt_residual_on_glacier']['data'][i, m-1] += np.sum(bias_on_g)
                out['liq_prcp_off_glacier']['data'][i, m-1] += np.sum(liq_prcp_off_g)
                out['liq_prcp_on_glacier']['data'][i, m-1] += np.sum(liq_prcp_on_g)
                out['snowfall_off_glacier']['data'][i, m-1] += np.sum(prcpsol_off_g)
                out['snowfall_on_glacier']['data'][i, m-1] += np.sum(prcpsol_on_g)
                out['avalanche_added']['data'][i, m-1] += np.sum(ava_added)
                out['avalanche_removed']['data'][i, m-1] += np.sum(ava_removed)

                # Snow bucket is a state variable - stored at end of timestamp
                if store_monthly_hydro:
                    if m == 12:
                        out['snow_bucket']['data'][i+1, 0] += np.sum(snow_bucket)
                    else:
                        out['snow_bucket']['data'][i, m] += np.sum(snow_bucket)
                else:
                    out['snow_bucket']['data'][i+1, m-1] += np.sum(snow_bucket)

        # Update the annual data
        out['off_area']['data'][i] = off_area_out
        out['on_area']['data'][i] = on_area_out

        # If monthly, put the residual where we can
        if store_monthly_hydro and mb_mod.bias != 0:
            for melt, bias in zip(
                    [
                        out['melt_on_glacier']['data'][i, :],
                        out['melt_off_glacier']['data'][i, :],
                    ],
                    [
                        out['melt_residual_on_glacier']['data'][i, :],
                        out['melt_residual_off_glacier']['data'][i, :],
                    ],
            ):

                real_melt = melt - bias
                to_correct = utils.clip_min(real_melt, 0)
                to_correct_sum = np.sum(to_correct)
                if (to_correct_sum > 1e-7) and (np.sum(melt) > 0):
                    # Ok we correct the positive melt instead
                    fac = np.sum(melt) / to_correct_sum
                    melt[:] = to_correct * fac

        if do_spinup and yr < start_dyna_model_yr:
            residual_mb = 0
            model_mb = (out['snowfall_on_glacier']['data'][i, :].sum() -
                        out['melt_on_glacier']['data'][i, :].sum())
        else:
            # Correct for mass-conservation and match the ice-dynamics model
            fmod.run_until(yr + 1)
            model_mb = (fmod.volume_m3 - prev_model_vol) * cfg.PARAMS['ice_density']
            prev_model_vol = fmod.volume_m3

            reconstructed_mb = (out['snowfall_on_glacier']['data'][i, :].sum() -
                                out['melt_on_glacier']['data'][i, :].sum())
            residual_mb = model_mb - reconstructed_mb

        # Now correct
        g_melt = out['melt_on_glacier']['data'][i, :]
        if residual_mb == 0:
            pass
        elif store_monthly_hydro:
            # We try to correct the melt only where there is some
            asum = g_melt.sum()
            if asum > 1e-7 and (residual_mb / asum < 1):
                # try to find a fac
                fac = 1 - residual_mb / asum
                # fac should always be positive, otherwise melt_on_glacier
                # gets negative. This is always true, because we only do the
                # correction if residual_mb / asum < 1
                corr = g_melt * fac
                residual_mb = g_melt - corr
                out['melt_on_glacier']['data'][i, :] = corr
            else:
                # We simply spread over the months
                residual_mb /= 12
                # residual_mb larger > melt_on_glacier
                # add absolute difference to snowfall (--=+, mass conservation)
                out['snowfall_on_glacier']['data'][i, :] -= utils.clip_max(g_melt - residual_mb, 0)
                # assure that new melt_on_glacier is non-negative
                out['melt_on_glacier']['data'][i, :] = utils.clip_min(g_melt - residual_mb, 0)
        else:
            # We simply apply the residual - no choice here
            # residual_mb larger > melt_on_glacier:
            # add absolute difference to snowfall (--=+, mass conservation)
            out['snowfall_on_glacier']['data'][i, :] -= utils.clip_max(g_melt - residual_mb, 0)
            # assure that new melt_on_glacier is non-negative
            out['melt_on_glacier']['data'][i, :] = utils.clip_min(g_melt - residual_mb, 0)

        out['model_mb']['data'][i] = model_mb
        out['residual_mb']['data'][i] = residual_mb

    # Convert to xarray
    out_vars = cfg.PARAMS['store_diagnostic_variables']
    ods = xr.Dataset()
    ods.coords['time'] = fmod.years
    if store_monthly_hydro:
        ods.coords['month_2d'] = ('month_2d', np.arange(1, 13))
        # For the user later
        sm = cfg.PARAMS['hydro_month_' + mb_mod.hemisphere]
        ods.coords['hydro_month_2d'] = ('month_2d', (np.arange(12) + 12 - sm + 1) % 12 + 1)
        ods.coords['calendar_month_2d'] = ('month_2d', np.arange(1, 13))
    for varname, d in out.items():
        data = d.pop('data')
        if varname not in out_vars:
            continue
        if len(data.shape) == 2:
            # First the annual agg
            if varname == 'snow_bucket':
                # Snowbucket is a state variable
                ods[varname] = ('time', data[:, 0])
            else:
                # Last year is never good
                data[-1, :] = np.nan
                ods[varname] = ('time', np.sum(data, axis=1))
            # Then the monthly ones
            if store_monthly_hydro:
                ods[varname + '_monthly'] = (('time', 'month_2d'), data)
        else:
            assert varname != 'snow_bucket'
            data[-1] = np.nan
            ods[varname] = ('time', data)
        for k, v in d.items():
            ods[varname].attrs[k] = v
            if store_monthly_hydro and (varname + '_monthly') in ods:
                ods[varname + '_monthly'].attrs[k] = v

    # Append the output to the existing diagnostics
    fpath = gdir.get_filepath('model_diagnostics', filesuffix=suffix)
    ods.to_netcdf(fpath, mode='a')
