"""This module allows the use of snowslide within the OGGM workflow

Version for OGGM 1.6.3.dev (tested with commit 27ba8ed65), with the daily,
surface-type (SfcType) and debris (EF_to_gdir) avalanche mass-balance models.
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
from numpy.typing import ArrayLike
import salem
import warnings

# Locals
from oggm import __version__
import oggm.cfg as cfg
from oggm import utils
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.exceptions import InvalidWorkflowError, InvalidParamsError
from oggm.core.massbalance import MassBalanceModel, mb_calibration_from_scalar_mb, decide_winter_precip_factor, MonthlyTIModel
from snowslide.snowslide_main import snowslide_base
from oggm.utils import (lazy_property, ncDataset, floatyear_to_date, date_to_floatyear, 
                        float_years_timeseries,
                        clip_min, clip_array)
from oggm.core.flowline import FileModel

# Climate relevant global params
MB_GLOBAL_PARAMS = [
    "temp_default_gradient",
    "temp_all_solid",
    "temp_all_liq",
    "temp_melt",
]


@utils.entity_task(log, writes=["gridded_data"])
def EF_to_gdir(gdir, path_to_EF):
    """Add the debris enhancement factor maps from Rounce et al. (2021) to this glacier directory

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process

    path_to_EF: string to directory with EF rasters
    """
    if path_to_EF is None:
        raise ValueError("Please provide a path_to_EF directory.")

    # Identify the file (handling both naming conventions)
    rgi_num = gdir.rgi_id.split('-')[-1]
    possible_names = [
        f"HMA_DTE_{rgi_num}_meltfactor_extrap.tif",
        f"HMA_DTE_{rgi_num}_meltfactor.tif"
    ]

    input_file = None
    for name in possible_names:
        fp = os.path.join(path_to_EF, name)
        if os.path.exists(fp):
            input_file = fp
            
            
    # Load and reproject
    if input_file is not None:
        # Open the GeoTiff
        dsb = salem.GeoTiff(input_file)
        
        # Salem's GeoTiff.get_vardata() handles the data reading
        # We read the first band (index 1)
        data = dsb.get_vardata(1)

        # If the raster has huge/tiny values representing NaNs, set them to 1.0
        # Rounce EF factors should realistically be between ~0.1 and 10.0
        data = np.where((data < 0) | (data > 100), np.nan, data)
        
        # Reproject using the gdir's grid (Millan-style)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning,
                                    message='.*out of bounds.*')
            # This maps the Rounce grid onto the OGGM grid
            r_data = gdir.grid.map_gridded_data(data, dsb.grid, interp='linear')
        
        # Convert to standard numpy and fill NaNs with 1.0 (neutral factor)
        # map_gridded_data returns a masked array
        total_ef = r_data.filled(1.0)

        # This catches the e+20 issue specifically
        total_ef = np.where(~np.isfinite(total_ef) | (total_ef > 20), 1.0, total_ef)
        total_ef = np.where(total_ef < 0, 1.0, total_ef)
    else:
        # If no file exists, default to 1.0 everywhere
        total_ef = np.ones(gdir.grid.ny_nx, dtype=np.float32)

    # Write
    with utils.ncDataset(gdir.get_filepath("gridded_data"), "a") as nc:
        vn = "EF_debris"

        # delete the variable if it already exists
        if vn in nc.variables:
            nc.remove_variable(vn)

        # create the variable
        v = nc.createVariable(vn, "f4", ("y", "x"), zlib=True)

        # set attributes
        v.units = 'unitless'
        v.long_name = 'Debris enhancement factors from Rounce et al. 2021'
        v.data_source = os.path.basename(input_file) if input_file else 'None'
        
        # Assign data
        v[:] = total_ef.astype(np.float32)


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


@utils.entity_task(log, fallback=_fallback)
def snowslide_statistics(gdir):
    """Gather statistics about the Snowslide snow redistribution"""

    try:
        # This is because the fallback doesnt have a return value (yet)
        resolution = abs(gdir.grid.dx)
    except:
        resolution = np.nan

    d = dict()
    # Easy stats - this should always be possible
    d["rgi_id"] = gdir.rgi_id
    d["rgi_region"] = gdir.rgi_region
    d["rgi_subregion"] = gdir.rgi_subregion
    d["rgi_area_km2"] = gdir.rgi_area_km2
    d["map_dx"] = resolution
    d["cenlat"] = gdir.cenlat
    d["cenlon"] = gdir.cenlon
    d["snowslide_1m_glacier_average"] = np.nan
    d["snowslide_deposit_area_km2"] = np.nan
    d["snowslide_total_accumulation_km3"] = np.nan
    d["snowslide_volume_added_km3"] = np.nan
    d["snowslide_volume_removed_km3"] = np.nan
    d["melt_f"] = gdir.read_json("mb_calib")['melt_f']
    d["temp_bias"] = gdir.read_json("mb_calib")['temp_bias']
    d["prcp_fac"] = gdir.read_json("mb_calib")['prcp_fac']
    d["melt_f_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['melt_f']
    d["temp_bias_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['temp_bias']
    d["prcp_fac_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['prcp_fac']

    try:
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            mask = ds["glacier_mask"]
            result = ds["snowslide_1m"].where(mask, np.nan)
            d["snowslide_1m_glacier_average"] = result.mean().data
            d["snowslide_deposit_area_km2"] = (
                float(result.where(result > 1, drop=True).count())
                * resolution**2 * 1e-6
            )
            if "snowslide_snd_before" in ds.variables:
                snd_before = ds["snowslide_snd_before"].where(mask, np.nan)
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
    """Monthly temperature index model with avalanches."""

    def __init__(
        self,
        gdir,
        filename: str = 'climate_historical',
        input_filesuffix: str = '',
        settings_filesuffix: str = '_with_ava',
        fl_id: int = None,
        melt_f: float = None,
        temp_bias: float = None,
        prcp_fac: float = None,
        bias: float = 0.0,
        temp_melt: float = None,
        ys: int = None,
        ye: int = None,
        repeat: bool = False,
        check_calib_params: bool = True,
        check_climate_data: bool = True,
        use_leap_years: bool = False,
    ):
        """Monthly temperature index model.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, default 'climate_historical'
            Set to a different BASENAME if you want to use alternative
            climate data.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        settings_filesuffix : str, optional
            Append a suffix to the settings file (useful for sensitivity
            runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            The value of the melt factor you want to use, here the unit
            is kg m-2 day-1 K-1. Defaults to the calibrated value.
        temp_bias : float, optional
            The value of the temperature bias. Defaults to the
            calibrated value.
        prcp_fac : float, optional
            The value of the precipitation factor. Defaults to the
            calibrated value.
        bias : float, default 0.0
            The value of the calibration bias [mm we yr-1]. Defaults to
            the calibrated value. Note that this bias is *subtracted*
            from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        temp_melt : float or None, default None
            The threshold for the air temperature above which ice melt is
            assumed to occur (-1°C the default for monthly mb models). If None
            settings['temp_melt'] is used.
        ys : int, optional
            The start of the climate period where the MB model is valid.
            Defaults to the period with available data.
        ye : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        repeat : bool, default False
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool, default True
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to ``False`` to suppress
            this check.
        check_climate_data : bool, default True
            If True the climate input data is checked if it is provided in total
            years and that the length matches.
        use_leap_years : bool, default False
            If the calendar should use leap years
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        super(MonthlyTIAvalancheModel, self).__init__(gdir=gdir,
                                             use_leap_years=use_leap_years)
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.fl_id = fl_id  # which flowline are we the model of?
        self.filename = filename
        self.input_filesuffix = input_filesuffix

        if melt_f is None:  # This prevents class methods
            melt_f = self.calib_params['melt_f']

        if temp_bias is None:
            temp_bias = self.calib_params['temp_bias']

        if prcp_fac is None:
            prcp_fac = self.calib_params['prcp_fac']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = self.calib_params['mb_global_params']
            for k, v in mb_calib.items():
                if v != self.gdir.settings[k]:
                    msg = ('You seem to use different mass balance parameters '
                           'than used for the calibration: '
                           f"you use gdir.settings['{k}']={gdir.settings[k]} while "
                           f"it was calibrated with gdir.settings['{k}']={v}. "
                           'Set `check_calib_params=False` to ignore this '
                           'warning.')
                    raise InvalidWorkflowError(msg)
            src = self.calib_params['baseline_climate_source']
            src_calib = gdir.get_climate_info(
                filename=self.filename, input_filesuffix=self.input_filesuffix
            )['baseline_climate_source']
            if src != src_calib:
                msg = (f'You seem to have calibrated with the {src} '
                       f"climate data while this gdir was calibrated with "
                       f"{src_calib}. Set `check_calib_params=False` to "
                       f"ignore this warning.")
                raise InvalidWorkflowError(msg)

        self.melt_f = melt_f
        self.bias = bias

        # Global parameters
        self.temp_all_solid = gdir.settings['temp_all_solid']
        self.temp_all_liq = gdir.settings['temp_all_liq']
        if temp_melt is None:
            self.temp_melt = gdir.settings['temp_melt']
        else:
            gdir.settings['temp_melt'] = temp_melt
            self.temp_melt = temp_melt

        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        self.temp_default_gradient = gdir.settings['temp_default_gradient']

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
        with ncDataset(fpath, mode='r') as nc:
            time = nc.variables["time"]
            time = cftime.num2date(time[:], time.units, calendar=time.calendar)

            # only use defined years
            years = np.array(list(map(lambda x: x.year, time)))
            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = years >= ys
            if ye is not None:
                try:
                    pok = pok & (years <= ye)
                except TypeError:
                    pok = years <= ye

            self.years = years[pok]
            self.months = np.array(list(map(lambda x: x.month, time)))[pok]
            self.days = np.array(list(map(lambda x: x.day, time)))[pok]

            if check_climate_data:
                # check for full years, this is overwritten for daily
                self._check_for_full_years()

            # Read timeseries and correct it
            self.temp = nc.variables["temp"][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables["prcp"][pok].astype(np.float64) * self._prcp_fac

            grad = self.prcp * 0 + self.temp_default_gradient
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.climate_source = nc.climate_source
            self.ys = self.years[0]
            self.ye = self.years[-1]
            self.ys_float = date_to_floatyear(self.years[0], self.months[0],
                                              self.days[0])
            self.ye_float = date_to_floatyear(self.years[-1], self.months[-1],
                                              self.days[-1])

    def __repr__(self):
        """String Representation of the mass balance model"""
        summary = ['<oggm.MassBalanceModel>']
        summary += ['  Class: ' + self.__class__.__name__]
        summary += ['  Attributes:']
        # Add all scalar attributes
        done = []
        for k in ['hemisphere', 'climate_source', 'melt_f', 'prcp_fac', 'temp_bias', 'bias']:
            done.append(k)
            v = self.__getattribute__(k)
            if k == 'climate_source':
                if v.endswith('.nc'):
                    v = os.path.basename(v)
            nofloat = ['hemisphere', 'climate_source']
            nbform = '    - {}: {}' if k in nofloat else '    - {}: {:.02f}'
            summary += [nbform.format(k, v)]
        for k, v in self.__dict__.items():
            if np.isscalar(v) and not k.startswith('_') and k not in done:
                nbform = '    - {}: {}'
                summary += [nbform.format(k, v)]
        return '\n'.join(summary) + '\n'

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: gdir.settings['prcp_fac'])

        Called factor to make clear that it is a multiplicative factor in
        contrast to the additive temperature bias
        """
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        # just to check that no invalid prcp_factors are used
        if np.any(np.asarray(new_prcp_fac) <= 0):
            raise InvalidParamsError('prcp_fac has to be above zero!')

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

    @lazy_property
    def calib_params(self):
        if self.fl_id is None:
            return self.gdir.settings
        else:
            fp_fl_settings = self.gdir.get_filepath('settings',
                                                    filesuffix=f'_fl{self.fl_id}')
            if os.path.exists(fp_fl_settings):
                self.gdir.settings_filesuffix = f'_fl{self.fl_id}'
                out = self.gdir.settings
                if self.settings_filesuffix:
                    raise InvalidWorkflowError('settings_filesuffix cannot be '
                                               'used with multiple flowlines')
                return out
            else:
                return self.gdir.settings

    def _check_for_full_years(self):
        # We check for full calendar years
        if self.years[0] != self.years[-1]:
            nr_of_months = (self.years[-1] - self.years[0] + 1) * 12
        else:
            nr_of_months = 12
        len_data_ok = len(self.years) == nr_of_months
        months_ok = (self.months[0] == 1) or (self.months[-1] == 12)
        if not months_ok or not len_data_ok:
            raise InvalidWorkflowError(
                "We now work exclusively with full calendar years. Check "
                "provided climate data! \n Your current selection: "
                f"{self.months[0]:02d}.{self.years[0]:04d} - "
                f"{self.months[-1]:02d}.{self.years[-1]:04d}\n"
                f"Your data has {len(self.years)} timestamps, but we expect "
                f"{nr_of_months}."
            )

    def is_year_valid(self, year: int) -> bool:
        """Check if a year is within the climate period.

        Returns
        -------
        bool
            True if the year is within the climate period.
        """
        return self.ys_float <= year <= self.ye_float

    def validate_year(self, year: int) -> int:
        """Get and validate if a year is outside the data's time range.

        Raises
        ------
        ValueError
            If the year is outside of the data's climate period.
        """
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(year):  # this is overloaded by subclasses
            raise ValueError(
                f'year {year} out of the valid time bounds: '
                f'[{self.ys_float}, {self.ye_float}]'
            )
        return year

    def _get_tempformelt(self, temp):
        tempformelt = temp - self.temp_melt
        clip_min(tempformelt, 0, out=tempformelt)
        return tempformelt

    def _get_prcpsol(self, prcp, temp):
        fac = 1 - (temp - self.temp_all_solid) / (self.temp_all_liq - self.temp_all_solid)
        return prcp * clip_array(fac, 0, 1)

    def _get_climate_for_index(self, heights: ArrayLike, pok: ArrayLike
                               ) -> tuple:
        """Returns climate information at provided heights and time indexes.

        If only one time index is provided also the climate information is
        returned in 1D, otherwise in 2D.

        Parameters
        ----------
        heights : array_like
            the heights of interest in meters
        pok : np.ndarray
            the time indexes of interest

        Returns
        -------
        tuple[np.ndarray]
            Temperature, melt temperatures, total precipitation, and
            solid precipitation for each height pixel.
        """
        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        heights = np.asarray(heights)  # sometimes heights are passed as lists
        npix = len(heights)

        if np.size(pok) == 1:
            # fast path for 1D data (one time index)
            # Compute temp and tempformelt (temperature above melting threshold)
            temp = itemp + igrad * (heights - self.ref_hgt)
            tempformelt = self._get_tempformelt(temp)

            # Compute solid precipitation from total precipitation
            prcp = np.ones(npix) * iprcp
            prcpsol = self._get_prcpsol(prcp, temp)

            return temp, tempformelt, prcp, prcpsol

        # otherwise we need to handle 2D data
        # Compute temp and tempformelt (temperature above melting threshold)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(len(pok)).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = self._get_tempformelt(temp2d)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        prcpsol = self._get_prcpsol(prcp, temp2d)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """
        y, m = floatyear_to_date(year)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok,)

        # get tmelt for the entire month
        tmelt *= self.days_in_month(year=year)

        # add precipitation correction factor from avalanching
        prcpsol = prcpsol * self.ava_prpc_fac

        return t, tmelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year=None):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        year = self.validate_year(year=year)
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok)

        return t, tmelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        tuple
            Mean temperature, and sums of melt temperature,
            precipitation, and solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(
            heights=heights, year=year)
        myr = date_to_floatyear(np.repeat(int(np.floor(year)), 12),
                                np.arange(1, 13))
        days_of_month = [self.days_in_month(year=yr) for yr in myr]
        # get tmelt for the entire month
        tmelt *= days_of_month

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self,
                       heights: np.ndarray,
                       year: float = None,
                       add_climate: bool = False,
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)

        # length of the month in days already considered in tmelt in get_monthly_climate
        mb_month = prcpsol - self.melt_f * tmelt
        sec_in_month = self.sec_in_month(year=year)
        mb_month -= (self.bias * sec_in_month / self.sec_in_year(year=year))
        rho = cfg.PARAMS['ice_density']
        if add_climate:
            return mb_month / sec_in_month / rho, t, tmelt, prcp, prcpsol
        return mb_month / sec_in_month / rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        """Get annual mass balance.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
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

        myr = date_to_floatyear(np.repeat(int(np.floor(year)), 12),
                                np.arange(1, 13))
        days_of_month = [self.days_in_month(year=yr) for yr in myr]
        # get tmelt for the entire month
        tmelt *= days_of_month

        mb_annual = np.sum(prcpsol - self.melt_f * tmelt, axis=1)
        rho = cfg.PARAMS['ice_density']
        mb_annual = ((mb_annual - self.bias) / self.sec_in_year(year=year) /
                     rho)
        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1), ava_added.sum(axis=1),
                    ava_removed.sum(axis=1))
        return mb_annual
    
    def get_annual_acc(self, heights, year=None):
        _, _, _, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)

        rho = cfg.PARAMS['ice_density']
        acc_annual = np.sum(prcpsol, axis=1)  / SEC_IN_YEAR / rho
        return acc_annual



class DailyTIAvalancheModel(MonthlyTIModel):
    """Daily temperature index model."""

    def __init__(
        self,
        gdir,
        filename: str = 'climate_historical_daily',
        input_filesuffix: str = '',
        settings_filesuffix: str = '_with_ava',
        fl_id: int = None,
        melt_f: float = None,
        temp_bias: float = None,
        prcp_fac: float = None,
        bias: float = 0.0,
        temp_melt: float = 0.0,
        ys: int = None,
        ye: int = None,
        repeat: bool = False,
        check_calib_params: bool = True,
        check_climate_data: bool = True,
        use_leap_years: bool = True,
    ):
        """Inherits from MonthlyTIModel.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, default 'climate_historical_daily'
            Set to a different BASENAME if you want to use alternative
            climate data.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        settings_filesuffix : str, optional
            Append a suffix to the settings file (useful for sensitivity
            runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            The value of the melt factor you want to use, here the unit
            is kg m-2 day-1 K-1. Defaults to the calibrated value.
        temp_bias : float, optional
            The value of the temperature bias. Defaults to the
            calibrated value.
        prcp_fac : float, optional
            The value of the precipitation factor. Defaults to the
            calibrated value.
        bias : float, default 0.0
            The value of the calibration bias [mm we yr-1]. Defaults to
            the calibrated value. Note that this bias is *subtracted*
            from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        temp_melt : float, default 0.0
            The threshold for the air temperature above which ice melt is
            assumed to occur (0°C the default for daily mb models).
        ys : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        ye : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        repeat : bool, default False
            Whether the climate period given by [ys, ye] should be
            repeated indefinitely in a circular way.
        check_calib_params : bool, default True
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to ``False`` to suppress
            this check.
        check_climate_data : bool, default True
            If True, check the climate input data is provided in total
            years and that the length matches.
        use_leap_years : bool, default True
            If the calendar should use leap years.
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        super(DailyTIAvalancheModel, self).__init__(
            gdir=gdir,
            filename=filename,
            input_filesuffix=input_filesuffix,
            settings_filesuffix=settings_filesuffix,
            fl_id=fl_id,
            melt_f=melt_f,
            temp_bias=temp_bias,
            prcp_fac=prcp_fac,
            bias=bias,
            temp_melt=temp_melt,
            ys=ys,
            ye=ye,
            repeat=repeat,
            check_calib_params=check_calib_params,
            check_climate_data=check_climate_data,
            use_leap_years=use_leap_years,
        )

        # Add avalanche data
        binned_data_file = gdir.get_filepath(
            "elevation_band_flowline", filesuffix="_fixed_dx"
        )
        binned_data = pd.read_csv(binned_data_file, index_col=0)
        self.ava_prpc_fac = binned_data.snowslide_1m.values

    def _check_for_full_years(self):
        # We check for full calendar years
        nr_of_days = 0
        for yr in np.arange(self.years[0], self.years[-1] + 1):
            nr_of_days += self.days_in_year(yr)
        len_data_ok = len(self.years) == nr_of_days
        months_ok = (self.months[0] == 1) and (self.months[-1] == 12)
        days_ok = (self.days[0] == 1) and (self.days[-1] == 31)
        if not months_ok or not days_ok or not len_data_ok:
            raise InvalidWorkflowError(
                "We now work exclusively with full calendar years (01.01. - "
                "31.12.). Check provided climate data!\nYour current selection: "
                f"{self.days[0]:02d}.{self.months[0]:02d}.{self.years[0]:04d} - "
                f"{self.days[-1]:02d}.{self.months[-1]:02d}.{self.years[-1]:04d}"
                f"\nYour data has {len(self.years)} timestamps, but we expect "
                f"{nr_of_days}."
            )

    def get_annual_climate(self, heights, year=None):
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def _get_2d_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        y, m = floatyear_to_date(year)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) & (self.months == m))[0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok)

        return t, tmelt, prcp, prcpsol

    def get_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """

        t, tmelt, prcp, prcpsol = self._get_2d_monthly_climate(heights, year)

        # add precipitation correction factor from avalanching
        ava_prpc_fac = self.ava_prpc_fac
        if len(heights) > len(ava_prpc_fac):
            padsize = len(heights) - len(ava_prpc_fac)
            ava_prpc_fac = np.pad(ava_prpc_fac, (0, padsize), constant_values=1.0)
        prcpsol = prcpsol * ava_prpc_fac

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_daily_climate(self,
                          heights: np.ndarray,
                          year: float = None,
                          ) -> tuple:
        """Daily climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """
        y, m, d = floatyear_to_date(year, return_day=True)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) &
                       (self.months == m) &
                       (self.days == d))[0][0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok, )
        
        # add precipitation correction factor from avalanching
        ava_prpc_fac = self.ava_prpc_fac
        if len(heights) > len(ava_prpc_fac):
            padsize = len(heights) - len(ava_prpc_fac)
            ava_prpc_fac = np.pad(ava_prpc_fac, (0, padsize), constant_values=1.0)
        prcpsol = prcpsol * ava_prpc_fac

        return t, tmelt, prcp, prcpsol

    def get_daily_mb(self,
                     heights: np.ndarray,
                     year: int = None,
                     add_climate: bool = False,
                     **kwargs,
                     ) -> np.float64 or tuple:
        """Get daily mass balance.

        Accounts for leap years by default.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : int, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Daily mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self.get_daily_climate(heights, year=year)

        mb_daily = prcpsol - self.melt_f * tmelt
        mb_daily -= (self.bias * SEC_IN_DAY / self.sec_in_year(year=year))
        rho = cfg.PARAMS['ice_density']

        if add_climate:
            return (mb_daily / SEC_IN_DAY / rho,
                    t, tmelt, prcp, prcpsol)
        return mb_daily / SEC_IN_DAY / rho

    def get_monthly_mb(self,
                       heights: np.ndarray,
                       year: float = None,
                       add_climate: bool = False,
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : np.ndarray
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclass implementations of this
            method.


        Returns
        -------
        np.float64 or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_monthly_climate(heights, year)

        mb_month = np.sum(prcpsol - self.melt_f * tmelt,
                          axis=1)
        sec_in_month = self.sec_in_month(year=year)
        mb_month -= (self.bias * sec_in_month / self.sec_in_year(year=year))
        rho = cfg.PARAMS['ice_density']

        if add_climate:
            return (mb_month / sec_in_month / rho, t.mean(axis=1),
                    tmelt.sum(axis=1), prcp.sum(axis=1), prcpsol.sum(axis=1))

        return mb_month / sec_in_month / rho

    def get_annual_mb(self,
                      heights: np.ndarray,
                      year: float = None,
                      add_climate: bool = False,
                      **kwargs,
                      ) -> np.float64 or tuple:
        """Get annual mass balance.

        This is equivalent to taking the sum of ``get_daily_mb``.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to ``get_2d_temperature``.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
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
        rho = cfg.PARAMS['ice_density']

        mb_annual = np.sum(prcpsol - self.melt_f * tmelt,
                           axis=1)
        mb_annual = ((mb_annual - self.bias) / self.sec_in_year(year=year) /
                     rho)

        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1), ava_added.sum(axis=1),
                    ava_removed.sum(axis=1))
        return mb_annual
    
    def get_annual_acc(self, heights, year=None):
        _, _, _, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)
        rho = cfg.PARAMS['ice_density']

        acc_annual = np.sum(prcpsol, axis=1)  / SEC_IN_YEAR / rho
        return acc_annual


class SfcTypeTIAvalancheModel(MassBalanceModel):
    """Temperature-Index model with surface type distinction using a bucket
    system.

    Adapted from OGGM/massbalance-sandbox and explained in Schuster et al. 2023.
    """

    def __init__(
        self,
        gdir,
        settings_filesuffix: str = "_with_ava",
        use_leap_years: bool = True,
        mb_model_class=DailyTIAvalancheModel,
        climate_resolution: str = "annual",
        aging_frequency: str = "annual",
        melt_f_ratio: float = 0.5,
        melt_f_change: str = "neg_exp",
        tau_e: float = 1.0,
        ys: int = 2000,
        spinup_years: int = 6,
        save_spinup_mbs: bool = False,
        spinup_buckets: np.ndarray = None,
        fl = None,
        use_main_fl_from: str = 'inversion_flowlines',
        hbins: ArrayLike = None,
        store_buckets: bool or str = False,
        store_buckets_dates: ArrayLike = None,
        use_previous_mbs: bool = False,
        store_snowline: bool = False,
        store_snowline_start_month: str = 'Oct',
        **kwargs,
    ):
        """Surface type temperature index model.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        settings_filesuffix : str, optional
            append a suffix to the settings file (useful for
            sensitivity runs).
        use_leap_years : bool, default True
            If the calendar should use leap years.
        mb_model_class : MassBalanceModel, default ``DailyTIModel``
            The MassBalanceModel to use together with surface type tracking.
            Options are DailyTIModel and MonthlyTIModel.
        climate_resolution : str, default 'annual'
            Temporal mass balance resolution. Options are 'annaul', 'monthly'
            and 'daily', but the mb_model_class need to be able to provide this
            resolution.
        aging_frequency : str, default "annual"
            Frequency at which the buckets are moved to the next one (= getting
            older). Options are "annual" or "monthly".
            If annual, the model uses one snow and (spinup_years-1) firn buckets.
            If monthly, the snow ages over the number of months * spinup_years.
        melt_f_ratio : float, default 0.5
            Ratio of snow melt factor to ice melt factor.
            Between 0 and 1, where 1 is no surface type distinction.
            Default 0.5 to match GloGEM.
        melt_f_change : str, default "neg_exp"
            How the snow melt factor changes relative to the ice melt
            factor, either "linear" or "neg_exp" (see `tau_e` for the equation).
        tau_e : float, default 1
            Only used if melt_f_change is 'neg_exp'. It describes how fast the
            snow melt factor approximates to the ice melt factor via
            melt_f=melt_f_ice+(melt_f_snow-melt_f_ice)*np.exp(-time_yr/tau_e_fold_yr)
            Must be larger than zero to prevent ``melt_f`` being set to NaN in
            the first bucket.
        ys : int, default 2000
            The initial year from where we want to get mb values. This means the
            bucket system is spun up so that at ys for the first time ice
            can form. For spinup the years ys - spinup_years up to ys are used.
        spinup_years : int, default 6
            Number of spinup years. This defines the number of buckets we use
            (see aging_frequency for explanation). The minimum allowed value is
            1 (corresponds to only having a snow bucket for
            aging_frequency='annual'). The spinup ensures every bucket has the
            opportunity to be filled and ice can form at ys.
        save_spinup_mbs : bool, default False
            If we want to safe the climatic mb and ice mb during the spinup
            period. This is mainly useful for testing.
        spinup_buckets : np.ndarray, default None
            Instead of conducting a spinup of the buckets you can provide the
            initial buckets. It is assumed that these buckets are valid at the
            time ys.
        fl: Flowline, default None
            The flowline for defining the grid-points. If None, the flowline
            defined in `use_main_fl_from` is used. During all consecutive calls
            it is always checked the length of provided heights equals the
            length of the grid-points of the flowline.
        use_main_fl_from: str, default 'inversion_flowlines'
            If the main flowline of 'inversion_flowlines' or 'model_flowlines'
            should be used. If fl is provided this is ignored.
        hbins: ArrayLike, default None
            Height bins for classifying surface types. Only needed for
            ``ConstantMBModel``. Currently untested!
        store_buckets: bool or str, default False
            If buckets should be stored along the way. Options are 'annual',
            'monthly', 'daily' or False.
        store_buckets_dates: ArrayLike, default []
            In addition to the frequency of when the buckets should be stored
            defined in `store_buckets` you can also provide a few selected dates
            where the buckets should be stored.
        use_previous_mbs: bool, default False
            If True, and you call get_annual_mb, get_monthly_mb or get_daily_mb
            it is checked if the desired year was already computed and return
            this values. In this case the provided heights will be ignored. This
            can be needed for some tasks (e.g. run_with_hydro), but the user
            should be aware that this behaviour is different than other
            mb_models work, because with surface tracking we have included a
            memory of the past. If False and you try to revisit a past year and
            error is raised.
        store_snowline: bool, default False
            Experimental: save the snowline in the climate_resolution. For this
            the lowest elevation where the bucket is not zero is set as the
            snowline. Older buckets are considered starting from
            store_snowline_start_month.
        store_snowline_start_month: str, default 'Oct'
            Experimental: the start month of buckets used for the derivation of
            the snowline. E.g. with the default 'Oct', if you want to derive the
            snowline at Mar, all buckets starting from the previous Oct are
            considered.
        **kwargs:
            keyword arguments to pass to the mb_model_class
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        super(SfcTypeTIAvalancheModel, self).__init__(gdir=gdir,
                                             use_leap_years=use_leap_years)

        # the mass balance model we use to get the climate data, as well as
        # melt_f, prcp_fac and temp_bias
        if mb_model_class not in [MonthlyTIAvalancheModel, DailyTIAvalancheModel, MonthlyTIAvalancheDebrisModel, DailyTIAvalancheDebrisModel]:
            raise NotImplementedError(f"mb_model_class: {mb_model_class}")
        self.mbmod = mb_model_class(gdir=gdir,
                                    settings_filesuffix=settings_filesuffix,
                                    **kwargs)
        self.filename = self.mbmod.filename
        self.input_filesuffix = self.mbmod.input_filesuffix
        self.hemisphere = self.mbmod.hemisphere
        self.bias = self.mbmod.bias
        self.ys = self.mbmod.ys
        self.ye = self.mbmod.ye

        # check compatibility of aging_frequency and climate_resolution: aging
        # can not happen at higher temporal resolution than the climate steps
        if aging_frequency not in ["annual", "monthly"]:
            raise NotImplementedError(f"aging_frequency. {aging_frequency}")
        climate_and_aging_compatible = True
        if climate_resolution == 'annual':
            if aging_frequency != 'annual':
                climate_and_aging_compatible = False
        elif climate_resolution in ['monthly', 'daily']:
            # currently the fastest aging supported is monthly
            pass
        else:
            raise NotImplementedError(
                f"climate_resolution: {climate_resolution}")
        if not climate_and_aging_compatible:
            raise InvalidParamsError(
                f"The selected climate_resolution '{climate_resolution}' is "
                f"not compatible with an aging_frequency '{aging_frequency}'. "
                "The climate resolution must be the same or shorter than "
                "aging (e.g. you can not age monthly while the climate is "
                "only applied annual).")
        self.aging_frequency = aging_frequency
        self.climate_resolution = climate_resolution

        # stuff related to spinup, only used if spinup_buckets is None
        if spinup_years < 1:
            raise InvalidParamsError("spinup_years needed to be at least 1, you "
                                     f"provided {spinup_years}.")
        self.spinup_years = spinup_years
        self.save_spinup_mbs = save_spinup_mbs
        self.spinup_buckets = spinup_buckets
        if isinstance(self.spinup_buckets, pd.core.frame.DataFrame):
            self.spinup_buckets = self.spinup_buckets.values
        self.ys = ys
        if self.spinup_buckets is None:
            # check if climate data for spinup is available
            spinup_start = self.ys - self.spinup_years
            try:
                self.mbmod.validate_year(spinup_start)
                self.mbmod.validate_year(self.ys)
            except ValueError as e:
                raise ValueError(
                    "Climate data for spinup not available. We need data for "
                    f"the period {spinup_start} (ys - spinup_years) to "
                    f"{self.ys}, but we get the following error: {e}")

        # defining the number of grid points and the spinup heights, either with
        # fl or hbins
        self.hbins = hbins  # TODO: need to test with constant_mb (maybe hbins[::-1])
        if fl is not None:
            self.fl = fl
        else:
            if use_main_fl_from == 'inversion_flowlines':
                self.fl = self.gdir.read_pickle("inversion_flowlines")[-1]
            elif use_main_fl_from == 'model_flowlines':
                self.fl = self.gdir.read_pickle("model_flowlines")[-1]
            elif self.hbins is not None:
                # ok we are working with height bins here, don't need the fl
                self.fl = None
            else:
                raise InvalidParamsError("We need a flowline or height bins "
                                         "(hbins) for defining the number of "
                                         "buckets we want to compute!")

        # create labels for the grid_points and define heights used during spinup
        if self.fl is not None:
            # for the flowline we use the distance along the flowline
            self.buckets_grid_point_label = self.fl.dx_meter * np.arange(self.fl.nx)
            self.spinup_heights = self.fl.surface_h
        else:
            # for hbins we just use numbers
            self.buckets_grid_point_label = np.arange(len(self.hbins))
            self.spinup_heights = self.hbins

        # here we set the names of the buckets, this depends on the
        # aging_frequency and spinup_years
        if self.aging_frequency == "annual":
            firn_buckets = [f"firn_{i}" for i in range(1, self.spinup_years, 1)]
        elif self.aging_frequency == "monthly":
            firn_buckets = [f"firn_{i}" for i in
                            range(1, 12 * self.spinup_years, 1)]
        else:
            raise InvalidParamsError("aging_frequency must be 'annual' or "
                                     "'monthly'")

        # we also add a snow and an ice bucket
        self.buckets = ["snow"] + firn_buckets + ["ice"]

        # set a template for an empty bucket and mb containers
        self._empty_mb_buckets_np = np.zeros((len(self.buckets_grid_point_label),
                                              len(self.buckets)))

        # stuff related to varying melt_f for each bucket
        self.tau_e = tau_e
        self.melt_f_ratio = melt_f_ratio
        self.melt_f_change = melt_f_change
        self.melt_f_buckets = None
        self.set_melt_f_buckets()

        # define if some additonal outputs should be saved along the way
        if ((store_buckets == 'monthly' and
             self.climate_resolution == 'annual') or
                (store_buckets == 'daily' and
                 self.climate_resolution != 'daily')):
            raise InvalidParamsError(
                "It is not possible to save the buckets in a higher resolution "
                "than the applied climate! Your selection: store_buckets = "
                f"{store_buckets}, climate_resolution = "
                f"{self.climate_resolution}")
        self.store_buckets = store_buckets
        if store_buckets_dates is not None:
            # if specific dates are provided this overrules store_buckets
            self.store_buckets = True
        self.store_buckets_dates = store_buckets_dates

        # if the user wants to return previously calculated mb values
        self.use_previous_mbs = use_previous_mbs

        # snowline stuff
        self.store_snowline = store_snowline
        if self.store_snowline:
            # how many buckets we need to add for the snowline buckets
            self._snowline_start_month = {
                "Jan": 11, "Feb": 10, "Mar": 9, "Apr": 8, "May": 7, "Jun": 6,
                "Jul": 5, "Aug": 4, "Sep": 3, "Oct": 2, "Nov": 1, "Dec": 0,
            }[store_snowline_start_month]
            self._snowline = []
            self._snowline_year = []
            # this are the height values for the special cases fully snow
            # covered or fully snow free
            self.snowline_inf_values = {}

        # Initialise buckets and conduct a potential spinup
        self._init_buckets()

    def _init_buckets(self):
        # reset some containers for a fresh start
        self.mb_buckets_np = self._empty_mb_buckets_np.copy()  # kg m-2
        # define length of needed timesteps
        first_year = self.ys
        if self.save_spinup_mbs:
            first_year -= self.spinup_years
        if self.climate_resolution == 'annual':
            self.nr_timesteps = len(range(first_year, self.ye + 1))
        elif self.climate_resolution == 'monthly':
            self.nr_timesteps = len(float_years_timeseries(
                y0=first_year, y1=self.ye, include_last_year=True, ))
        elif self.climate_resolution == 'daily':
            self.nr_timesteps = len(float_years_timeseries(
                y0=first_year, y1=self.ye, include_last_year=True,
                daily=True))
        else:
            raise NotImplementedError(
                f"climate_resolution {self.climate_resolution}")

        output_shape = (self.nr_timesteps,  # total nr of timesteps
                        len(self.buckets_grid_point_label))  # nr of grid points
        self._climatic_mb = np.empty(output_shape)  # kg m-2
        self._ice_mb = np.empty(output_shape)  # kg m-2
        self._mb_heights = np.empty(output_shape)  # m
        self._year_to_index = {}  # saving the array positions of years
        self._current_index = 0  # keep track of last added position

        if self.store_buckets:
            # the mb_buckets are stored in a dict, with key corresponding to the
            # date and the buckets saved as pandas dataframes
            self.mb_buckets_stored = {}

        # now define the initial buckets
        if self.spinup_buckets is not None:
            # ok user provided the initial buckets, check shape and use
            provided_shape = self.spinup_buckets.shape
            needed_shape = self._empty_mb_buckets_np.shape
            if (provided_shape[0] != needed_shape[0] or
                    provided_shape[1] != needed_shape[1] - 1):  # exclude ice
                raise InvalidParamsError(
                    "The provided spinup_buckets have the shape "
                    f"{provided_shape}, but we expect the shape "
                    f"({needed_shape[0]}, {needed_shape[1] - 1}).")

            self.mb_buckets_np = self._empty_mb_buckets_np.copy()
            self.mb_buckets_np[:, :-1] = self.spinup_buckets.copy()
            self.mb_buckets_year = self.ys

        else:
            # we need to make a spinup
            # define the spinup timesteps depending on the climate resolution
            spinup_start_year = self.ys - self.spinup_years
            if self.climate_resolution == 'annual':
                spinup_steps = range(spinup_start_year, self.ys)
            elif self.climate_resolution == 'monthly':
                spinup_steps = float_years_timeseries(
                    y0=spinup_start_year, y1=self.ys)[:-1]
            elif self.climate_resolution == 'daily':
                spinup_steps = float_years_timeseries(
                    y0=spinup_start_year, y1=self.ys, daily=True)[:-1]
            else:
                raise NotImplementedError(
                    f"climate_resolution {self.climate_resolution}")

            # do the actual spinup
            for yr in spinup_steps:
                self._apply_climate_step_and_aging_to_buckets(
                    heights=self.spinup_heights, year=yr,
                    save_mbs=self.save_spinup_mbs)

            assert self.mb_buckets_year == self.ys

    @property
    def mb_buckets(self):
        # returns current snow and firn buckets, but not the ice bucket,
        # for ice look at ice_mb
        return pd.DataFrame(self.mb_buckets_np[:, :-1],
                            index=self.buckets_grid_point_label,
                            columns=self.buckets[:-1],)

    @property
    def climatic_mb(self):
        pd_dict = {}
        for year in self._year_to_index:
            pd_dict[year] = self._climatic_mb[[self._year_to_index[year]]][0]
        return pd.DataFrame(pd_dict,
                            index=self.buckets_grid_point_label)

    @property
    def ice_mb(self):
        pd_dict = {}
        for year in self._year_to_index:
            pd_dict[year] = self._ice_mb[[self._year_to_index[year]]][0]
        return pd.DataFrame(pd_dict,
                            index=self.buckets_grid_point_label)

    @property
    def mb_heights(self):
        pd_dict = {}
        for year in self._year_to_index:
            pd_dict[year] = self._mb_heights[[self._year_to_index[year]]][0]
        return pd.DataFrame(pd_dict,
                            index=self.buckets_grid_point_label)

    @property
    def snowline(self):
        return np.array(self._snowline)

    @property
    def snowline_year(self):
        return np.array(self._snowline_year)

    @property
    def melt_f(self):
        return self.mbmod.melt_f

    @melt_f.setter
    def melt_f(self, value):
        """Set new melt_f and reset the buckets."""
        self.mbmod.melt_f = value
        self.set_melt_f_buckets()
        # Recompute buckets
        self._init_buckets()

    @property
    def prcp_fac(self):
        """Precipitation factor"""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Set new precipitation factor and reset buckets."""
        self.mbmod.prcp_fac = value
        # Recompute buckets
        self._init_buckets()

    @property
    def temp_bias(self):
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Set new temperature bias and reset buckets."""
        self.mbmod.temp_bias = value
        # Recompute buckets
        self._init_buckets()

    def set_melt_f_buckets(self):
        """Set the melt factor for each bucket."""
        if self.melt_f_change == "linear":
            self.melt_f_buckets = dict(
                zip(self.buckets,
                    np.linspace(self.melt_f * self.melt_f_ratio, self.melt_f,
                                len(self.buckets),),
                    )
            )
        elif self.melt_f_change == "neg_exp":
            if self.tau_e <= 0:
                raise InvalidParamsError("`tau_e` must be above zero for"
                                         "`melt_f_change` = 'neg_exp'.")
            buckets_linspace = np.linspace(0, self.spinup_years,
                                           len(self.buckets))
            melt_f_snow = self.melt_f_ratio * self.melt_f
            self.melt_f_buckets = dict(
                zip(self.buckets,
                    self.melt_f + (melt_f_snow - self.melt_f) *
                    np.exp(-buckets_linspace / self.tau_e),
                    )
            )
        else:
            raise NotImplementedError(f"melt_f_change: {self.melt_f_change}")

    def _apply_climate_step_and_aging_to_buckets(
        self,
        heights: ArrayLike,
        year: float = None,
        save_mbs: bool = True,
    ):
        """ This is the central method where the magic is happening.

        It gets the climate input data for the provided year, adds fresh snow to
        the snow bucket, and melts the buckets in the correct order. At the end
        it checks if aging is needed, which moves everything into the next older
        bucket.

        Parameters
        ----------

        heights: array_like
           heights used to get the climate input data, need to be the same
           length as the number of defined grid points and in the same order.
           The number of grid points was defined either through fl or hbins at
           initialisation.
        year: float
           year in calendar float year
        save_mbs: bool, default True
            If True, the climatic_mb and the ice_mb are saved. This is set to
            False during spinup

        """

        # Need to provide heights for the same grid points as the bucket was
        # initialized. They also need to be in the same order, but this we can
        # not check
        if len(heights) != len(self.buckets_grid_point_label):
            raise InvalidParamsError(f'Number of heights ({len(heights)}) does '
                                     'not match number of grid points provided '
                                     'on initialization '
                                     f'({len(self.buckets_grid_point_label)}).')

        # get the current buckets, first is snow, last is ice, inbetween firn
        mb_buckets_np = self.mb_buckets_np

        # get the climate data in the defined resolution
        if self.climate_resolution == "annual":
            t, tmelt, prcp, prcpsol = self.mbmod.get_annual_climate(heights, year)
        elif self.climate_resolution == "monthly":
            t, tmelt, prcp, prcpsol = self.mbmod.get_monthly_climate(heights, year)
        elif self.climate_resolution == "daily":
            t, tmelt, prcp, prcpsol = self.mbmod.get_daily_climate(heights, year)
        else:
            raise NotImplementedError(
                f"climate_resolution: {self.climate_resolution}")

        # now we use tmelt for melting the buckets in order, the following
        # algorithm is optimized for np arrays smaller to shape (1000, 72),
        # which corresponds to 1000 flowline grid points and 72 buckets (= 6
        # years with monthly aging). We do this only for the snow and firn
        # buckets and deal with ice afterwards. For checking the code I add the
        # shape of variables as comments.

        # here we will save the results, use view to modify in-place
        snow_buckets_new = mb_buckets_np[:, :-1]
        nr_grid_points, nr_buckets = snow_buckets_new.shape
        # we need the sum of the old buckets later for calculating delta kg m-2
        snow_buckets_old_sum = snow_buckets_new.sum(axis=1)
        # melt_f per bucket without ice, (1, nr_buckets)
        melt_f_buckets = np.asarray(list(self.melt_f_buckets.values())[:-1],
                                    dtype=float)[None, :]
        # add one axis to tmelt for correct shape, (nr_grid_points, 1)
        tmelt = tmelt[:, None]

        # all solid precip. goes into fresh snow bucket before we start melting
        snow_buckets_new[:, 0] += prcpsol

        # now calculate cumulative tfm needed to melt each bucket and subtract
        # available tmelt, finally we convert back to mass in each bucket
        # (nr_grid_points, nr_buckets)
        buckets_kg_m2_cumsum_left = (((snow_buckets_new / melt_f_buckets
                                       ).cumsum(axis=1) - tmelt) *
                                     melt_f_buckets)

        # when the cumsum is negative this bucket and all above have melted
        # completely, (nr_grid_points, nr_buckets)
        melted_buckets = buckets_kg_m2_cumsum_left < 0

        # number of fully melted buckets per grid point, (nr_grid_points,)
        nr_melted_buckets = melted_buckets.sum(axis=1)

        # set melted buckets to zero
        snow_buckets_new[melted_buckets] = 0.0

        # set buckets which have only partially melted
        some_left_buckets = np.where(nr_melted_buckets < nr_buckets)[0]
        if some_left_buckets.size:
            partly_melted_bucket = nr_melted_buckets[some_left_buckets]
            # Set the partially melted bucket to the leftover mass
            snow_buckets_new[some_left_buckets, partly_melted_bucket] = (
                buckets_kg_m2_cumsum_left[some_left_buckets, partly_melted_bucket])

        # Now deal with ice, ice melts where some tmelt is still left after
        # melting all buckets
        all_melted_grid_points = (nr_melted_buckets == nr_buckets)
        if np.any(all_melted_grid_points):
            remaining_tfm = -(buckets_kg_m2_cumsum_left[all_melted_grid_points, -1] /
                              melt_f_buckets[:, -1])
            ice_melt_kg_m2 = remaining_tfm * self.melt_f_buckets['ice']
            # we use -= here because there could be some newly formed ice
            # already in the ice bucket after aging
            self.mb_buckets_np[all_melted_grid_points, -1] -= ice_melt_kg_m2
        else:
            ice_melt_kg_m2 = None

        if save_mbs:
            # Calculate the climatic mb as delta of buckets and minus potential
            # ice melt, all in kg m-2
            delta_kg_m2 = (snow_buckets_new.sum(axis=1)  # sum of buckets at the end
                           - snow_buckets_old_sum)  # sum of buckets at the start
            if ice_melt_kg_m2 is not None:
                delta_kg_m2[all_melted_grid_points] -= ice_melt_kg_m2  # ice melt

            # save the climatic mb of this timestep
            self._climatic_mb[self._current_index] = delta_kg_m2

        # save the mb of ice, this includes potential ice gain from aging after
        # the call of _bucket_aging and melt where all snow/firn buckets are
        # empty
        if save_mbs:
            self._ice_mb[self._current_index] = self.mb_buckets_np[:, -1].copy()
        # we empty the ice bucket after saving to avoid any double counting
        self.mb_buckets_np[:, -1] = 0.0

        if save_mbs:
            self._mb_heights[self._current_index] = heights

        # at the end we save the year of current index and increase it
        if save_mbs:
            self._year_to_index[year] = self._current_index
            self._current_index += 1

        # update the current year of the buckets, this is set one timestep later
        # to the currently applied climate step (e.g. after applying the climate
        # of January the bucket refers to the first of February)
        if self.climate_resolution == 'annual':
            buckets_yr = int(year) + 1
            buckets_month = 1
            buckets_day = 1
            self.mb_buckets_year = buckets_yr
        elif self.climate_resolution == 'monthly':
            buckets_yr, buckets_month = floatyear_to_date(float(year))
            buckets_day = 1
            # special case when a new year starts
            if buckets_month == 12:
                buckets_yr += 1
                buckets_month = 1
            else:
                buckets_month += 1
            self.mb_buckets_year = date_to_floatyear(buckets_yr, buckets_month)
        elif self.climate_resolution == 'daily':
            buckets_yr, buckets_month, buckets_day = floatyear_to_date(
                float(year), return_day=True)
            # date_to_floatyear can deal with monthly and yearly overflows
            # (e.g. 32.01. == 01.02., and 32.12.2000 == 01.01.2001)
            self.mb_buckets_year = date_to_floatyear(
                buckets_yr, buckets_month, buckets_day + 1)
            # finally we get the values without the overflow for aging below
            buckets_yr, buckets_month, buckets_day = floatyear_to_date(
                self.mb_buckets_year, return_day=True
            )
        else:
            raise NotImplementedError(
                f"climate_resolution: {self.climate_resolution} ")

        # Finally the bucket aging, potentially new formed ice is considered in
        # the next climate step
        if self.aging_frequency == 'annual':
            if self.climate_resolution == 'annual':
                # we need to age each step
                self._bucket_aging()
            elif self.climate_resolution == 'monthly':
                # we age if the bucket month is January (bucket year is always
                # one timestep ahead of current climate year, see above)
                if buckets_month == 1:
                    self._bucket_aging()
            elif self.climate_resolution == 'daily':
                if buckets_month == 1 and buckets_day == 1:
                    self._bucket_aging()
            else:
                raise NotImplementedError(
                    f"climate_resolution: {self.climate_resolution} ")
        elif self.aging_frequency == 'monthly':
            if self.climate_resolution == 'monthly':
                # we need to age each step
                self._bucket_aging()
            elif self.climate_resolution == 'daily':
                if buckets_day == 1:
                    self._bucket_aging()
            else:
                # it is already checked at initialisation that
                # climate_resolution is <= aging_frequency, but if we have
                # overseen something, this should raise
                raise NotImplementedError(
                    f"climate_resolution: {self.climate_resolution} ")
        else:
            raise NotImplementedError(
                f"aging_frequency: {self.aging_frequency} ")

        # check if the resulting bucket should be stored for later
        if self.store_buckets:
            store_current_bucket = False
            if self.store_buckets_dates is not None:
                if self.mb_buckets_year in self.store_buckets_dates:
                    store_current_bucket = True
            elif self.store_buckets == 'annual':
                if buckets_month == 1 and buckets_day == 1:
                    store_current_bucket = True
            elif self.store_buckets == 'monthly':
                if buckets_day == 1:
                    store_current_bucket = True
            elif self.store_buckets == 'daily':
                store_current_bucket = True
            else:
                raise NotImplementedError(f"store_buckets: {self.store_buckets}")

            if store_current_bucket:
                self.mb_buckets_stored[self.mb_buckets_year] = pd.DataFrame(
                    self.mb_buckets_np[:, :-1].copy(),
                    index=self.buckets_grid_point_label,
                    columns=self.buckets[:-1],)

        # store snowline
        if self.store_snowline:
            if self.mb_buckets_year not in self._snowline_year:
                number_buckets = buckets_month + self._snowline_start_month
                # + 1 is for the snow bucket
                not_melted_layers = np.any(
                    self.mb_buckets_np[:, :number_buckets + 1] > 0, axis=1)

                # check if completely snow free
                if not np.any(not_melted_layers):
                    self._snowline.append(np.inf)
                    # store a height value for snow free case
                    if np.inf not in self.snowline_inf_values:
                        self.snowline_inf_values[np.inf] = np.max(heights) + 50
                # check if fully snow covered
                elif np.all(not_melted_layers):
                    self._snowline.append(-np.inf)

                    if -np.inf not in self.snowline_inf_values:
                        # store a height value for fully snow covered case
                        self.snowline_inf_values[-np.inf] = np.min(heights) - 50
                # otherwise get the lowest elevation band with snow cover
                else:
                    self._snowline.append(heights[not_melted_layers][-1])
                self._snowline_year.append(self.mb_buckets_year)

        # nothing to return as every thing is stored is some variables
        return None

    def _bucket_aging(self):
        """In this function the aging of buckets happens.

        This means all buckets are moved one step and an empty fresh snow bucket
        is added at the top. The last bucket is the ice bucket and it is always
        set to zero before calling the aging in
        _apply_climate_step_and_aging_to_buckets (but we also check for this).
        """

        mb_buckets_np = self.mb_buckets_np

        # check that all snow and firn buckets are positive
        if mb_buckets_np[:, :-1].min() < 0:
            raise ValueError("Bucket values must all be positive.")

        # check ice bucket is empty and we do not loose any mass
        if np.any(mb_buckets_np[:, -1]):
            raise ValueError("Ice bucket not empty, need to check!")

        mb_buckets_val_old = mb_buckets_np[:, :-1]
        len_h = len(self.buckets_grid_point_label)
        # add a new fresh snow bucket and move everything to the next older
        # bucket, the last firn bucket is transferred to ice here
        updated_bucket_np = np.concatenate(
            [np.zeros(len_h).reshape(len_h, 1), mb_buckets_val_old], axis=1)

        self.mb_buckets_np = updated_bucket_np

    def _run_until(self, heights, year, mb_resolution):

        # check the current year of the bucket has not exceeded the current year
        if self.mb_buckets_year > year and not self.use_previous_mbs:
            raise InvalidWorkflowError(
                "The current buckets are valid for the float year "
                f"{self.mb_buckets_year}, and you want to have values for "
                f"{year}. If you want to get previously calculated mass "
                "balance values set use_previous_mbs to True. Be aware that "
                "in case that previously calculated mb values are available "
                "your provided heights here will be ignored!")

        # calculate all needed time steps with the same heights
        if self.climate_resolution == 'annual':
            if mb_resolution == 'annual':
                missing_float_years = range(self.mb_buckets_year, int(year) + 1)
            else:
                # mb_resolution can not be shorter than climate_resolution
                raise NotImplementedError(f"mb_resolution: {mb_resolution}")
        elif self.climate_resolution == 'monthly':
            if mb_resolution == 'annual':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=year,
                    include_last_year=True, )
            elif mb_resolution == 'monthly':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=np.ceil(year) + 1)
                # only keep those actually needed
                missing_float_years = [yr for yr in missing_float_years
                                       if yr <= year]
            else:
                # mb_resolution can not be shorter than climate_resolution
                raise NotImplementedError(f"mb_resolution: {mb_resolution}")
        elif self.climate_resolution == 'daily':
            if mb_resolution == 'annual':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=year,
                    include_last_year=True, daily=True)
            elif mb_resolution == 'monthly':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=np.ceil(year) + 1,
                    daily=True
                )
                # only keep those actually needed
                y, m = floatyear_to_date(year)
                y_end = y if m != 12 else y + 1
                m_end = m + 1 if m != 12 else 1
                missing_float_years = [yr for yr in missing_float_years
                                       if yr < date_to_floatyear(y_end, m_end)]
            elif mb_resolution == 'daily':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=np.ceil(year) + 1,
                    daily=True
                )
                # only keep those actually needed
                y, m, d = floatyear_to_date(year, return_day=True)
                missing_float_years = [yr for yr in missing_float_years
                                       # date_to_floatyear can handle overflows
                                       # e.g. 32.12.2000 == 01.01.2001
                                       if yr < date_to_floatyear(y, m, d + 1)]
            else:
                raise NotImplementedError(f"mb_resolution: {mb_resolution}")
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # if use_previous_mbs=True it could be there are no missing_float_years
        for yr in missing_float_years:
            if yr < self.mb_buckets_year:
                continue

            self._apply_climate_step_and_aging_to_buckets(heights=heights,
                                                          year=yr)

    def get_annual_mb(self,
                      heights: ArrayLike,
                      year: int or float,
                      add_climate: bool = False,
                      climatic_mb_or_ice_mb: str = 'climatic_mb',
                      **kwargs,
                      ) -> np.float64 or tuple:
        """Get annual climatic mass balance or the ice mass balance.

        The ice mass balance is important for ice dynamics.

        Parameters
        ----------
        heights : ArrayLike
            The heights of each grid point for the provided year. They need to
            have the same length as the number of defined grid points and in
            the same order. The number of grid points at its order were defined
            either through fl or hbins at initialisation.
        year : int or float
            Year in calendar float year or as int
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        climatic_mb_or_ice_mb : str, default 'climatic_mb'
            Defines if you want to retrive the climatic mass balance or the mass
            balance only for ice. The later one is meant to be used together
            with ice dynamics
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual climatic mass balance or ice mass balance in metres of ice
            per second. If ``add_climate`` is True, also returns mean
            temperature, the sums of melt temperature, total precipitation, and
            solid precipitation.
        """

        # compute all steps up to the desired target year using constant heights
        self._run_until(heights=heights, year=year,
                        mb_resolution='annual')

        if climatic_mb_or_ice_mb == 'climatic_mb':
            mbs = self._climatic_mb
        elif climatic_mb_or_ice_mb == 'ice_mb':
            mbs = self._ice_mb
        else:
            raise NotImplementedError(
                f"'climatic_mb_or_ice_mb': {climatic_mb_or_ice_mb}")

        # ok now everything should be available, and we can sum up annual values
        # as needed
        if self.climate_resolution == 'annual':
            annual_mb = mbs[self._year_to_index[year]]
        elif self.climate_resolution == 'monthly':
            float_months = float_years_timeseries(y0=year, y1=year+1,
                                                  daily=False)[:-1]
            idx = [self._year_to_index[yr] for yr in float_months]
            annual_mb = np.sum(mbs[idx], axis=0)
        elif self.climate_resolution == 'daily':
            float_days = float_years_timeseries(y0=year, y1=year+1,
                                                daily=True)[:-1]
            idx = [self._year_to_index[yr] for yr in float_days]
            annual_mb = np.sum(mbs[idx], axis=0)
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # convert from kg m-2 to m s-1
        rho = cfg.PARAMS['ice_density']
        annual_mb = ((annual_mb - self.mbmod.bias) / self.sec_in_year(year) /
                     rho)

        if add_climate:
            # because of the use of different climate resolutions we always need
            # to make a distinct call here, even it is potentially already
            # called in _apply_climate_step_and_aging_to_buckets
            t, tmelt, prcp, prcpsol = self.mbmod.get_annual_climate(
                heights=self.mb_heights[year], year=year)
            return annual_mb, t, tmelt, prcp, prcpsol
        else:
            return annual_mb

    def get_monthly_mb(self,
                       heights: ArrayLike,
                       year: float,
                       add_climate: bool = False,
                       climatic_mb_or_ice_mb: str = 'climatic_mb',
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly climatic mass balance or the ice mass balance.

        The ice mass balance is important for ice dynamics.

        Parameters
        ----------
        heights : ArrayLike
            The heights of each grid point for the provided year. They need to
            have the same length as the number of defined grid points and in
            the same order. The number of grid points at its order were defined
            either through fl or hbins at initialisation.
        year : float
            Year in calendar float year
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        climatic_mb_or_ice_mb : str, default 'climatic_mb'
            Defines if you want to retrive the climatic mass balance or the mass
            balance only for ice. The later one is meant to be used together
            with ice dynamics
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Monthly climatic mass balance or ice mass balance in metres of ice
            per second. If ``add_climate`` is True, also returns mean
            temperature, the sums of melt temperature, total precipitation, and
            solid precipitation.
        """

        # compute all steps up to the desired target year using constant heights
        self._run_until(heights=heights, year=year,
                        mb_resolution='monthly')

        if climatic_mb_or_ice_mb == 'climatic_mb':
            mbs = self._climatic_mb
        elif climatic_mb_or_ice_mb == 'ice_mb':
            mbs = self._ice_mb
        else:
            raise NotImplementedError(
                f"'climatic_mb_or_ice_mb': {climatic_mb_or_ice_mb}")

        # ok now everything should be available, and we can sum up monthly
        # values as needed
        if self.climate_resolution == 'annual':
            raise NotImplementedError('You can not get a monthly mb with an '
                                      'annual climate resolution!')
        elif self.climate_resolution == 'monthly':
            monthly_mb = mbs[self._year_to_index[year]]
        elif self.climate_resolution == 'daily':
            y_start, m_start, d_start = floatyear_to_date(year,
                                                          return_day=True)
            float_days = float_years_timeseries(
                y0=year, y1=year + 1, daily=True)[:-1]
            # only keep days larger equal than the start of the month
            float_days = [day for day in float_days
                          if day >= date_to_floatyear(y_start, m_start, 1)]
            # only keep days smaller the start of the next month
            m_end = m_start + 1 if m_start != 12 else 1
            y_end = y_start if m_start != 12 else y_start + 1
            float_days = [day for day in float_days
                          if day < date_to_floatyear(y_end, m_end, 1)]

            idx = [self._year_to_index[yr] for yr in float_days]
            monthly_mb = np.sum(mbs[idx], axis=0)
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # convert from kg m-2 to m s-1
        rho = cfg.PARAMS['ice_density']
        monthly_mb = ((monthly_mb / self.sec_in_month(year=year) -
                       self.mbmod.bias / self.sec_in_year(year)) /
                      rho)

        if add_climate:
            # because of the use of different climate resolutions we always need
            # to make a distinct call here, even it is potentially already
            # called in _apply_climate_step_and_aging_to_buckets
            t, tmelt, prcp, prcpsol = self.mbmod.get_monthly_climate(
                heights=self.mb_heights[year], year=year)
            return monthly_mb, t, tmelt, prcp, prcpsol
        else:
            return monthly_mb

    def get_daily_mb(self,
                     heights: ArrayLike,
                     year: float,
                     add_climate: bool = False,
                     climatic_mb_or_ice_mb: str = 'climatic_mb',
                     **kwargs,
                     ) -> np.float64 or tuple:
        """Get daily climatic mass balance or the ice mass balance.

        The ice mass balance is important for ice dynamics.

        Parameters
        ----------
        heights : ArrayLike
            The heights of each grid point for the provided year. They need to
            have the same length as the number of defined grid points and in
            the same order. The number of grid points at its order were defined
            either through fl or hbins at initialisation.
        year : float
            Year in calendar float year
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        climatic_mb_or_ice_mb : str, default 'climatic_mb'
            Defines if you want to retrive the climatic mass balance or the mass
            balance only for ice. The later one is meant to be used together
            with ice dynamics
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Daily climatic mass balance or ice mass balance in metres of ice
            per second. If ``add_climate`` is True, also returns mean
            temperature, the sums of melt temperature, total precipitation, and
            solid precipitation.
        """

        # compute all steps up to the desired target year using constant heights
        self._run_until(heights=heights, year=year,
                        mb_resolution='daily')

        if climatic_mb_or_ice_mb == 'climatic_mb':
            mbs = self._climatic_mb
        elif climatic_mb_or_ice_mb == 'ice_mb':
            mbs = self._ice_mb
        else:
            raise NotImplementedError(
                f"'climatic_mb_or_ice_mb': {climatic_mb_or_ice_mb}")

        # ok now everything should be available
        if self.climate_resolution in ['annual', 'monthly']:
            raise NotImplementedError(
                'You can not get a daily mb with an annual or monthly climate '
                f'resolution! Your climate resolution: {self.climate_resolution}')
        elif self.climate_resolution == 'daily':
            daily_mb = mbs[self._year_to_index[year]]
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # convert from kg m-2 to m s-1
        rho = cfg.PARAMS['ice_density']
        daily_mb = ((daily_mb / SEC_IN_DAY -
                     self.mbmod.bias / self.sec_in_year(year)) /
                    rho)

        if add_climate:
            # because of the use of different climate resolutions we always need
            # to make a distinct call here, even it is potentially already
            # called in _apply_climate_step_and_aging_to_buckets
            t, tmelt, prcp, prcpsol = self.mbmod.get_daily_climate(
                heights=self.mb_heights[year], year=year)
            return daily_mb, t, tmelt, prcp, prcpsol
        else:
            return daily_mb

    def is_year_valid(self, year):
        return self.mbmod.is_year_valid(year)

    def get_ela(self, year=None, **kwargs):
        # an idea to implement get_ela could be to use the previously computed
        # climatic_mb values together with the used heights and do an
        # interpolation to find the height where climatic_mb = 0, see spline_fit
        # in dynamic spinup
        raise NotImplementedError("Getting the ela for SfcTypeTIModel is"
                                  "currently not supported.")


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





class MonthlyTIAvalancheDebrisModel(MassBalanceModel):
    """Monthly temperature index model with avalanches & debris."""

    def __init__(
        self,
        gdir,
        filename: str = 'climate_historical',
        input_filesuffix: str = '',
        settings_filesuffix: str = '_with_avadeb',
        fl_id: int = None,
        melt_f: float = None,
        temp_bias: float = None,
        prcp_fac: float = None,
        bias: float = 0.0,
        temp_melt: float = None,
        ys: int = None,
        ye: int = None,
        repeat: bool = False,
        check_calib_params: bool = True,
        check_climate_data: bool = True,
        use_leap_years: bool = False,
    ):
        """Monthly temperature index model.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, default 'climate_historical'
            Set to a different BASENAME if you want to use alternative
            climate data.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        settings_filesuffix : str, optional
            Append a suffix to the settings file (useful for sensitivity
            runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            The value of the melt factor you want to use, here the unit
            is kg m-2 day-1 K-1. Defaults to the calibrated value.
        temp_bias : float, optional
            The value of the temperature bias. Defaults to the
            calibrated value.
        prcp_fac : float, optional
            The value of the precipitation factor. Defaults to the
            calibrated value.
        bias : float, default 0.0
            The value of the calibration bias [mm we yr-1]. Defaults to
            the calibrated value. Note that this bias is *subtracted*
            from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        temp_melt : float or None, default None
            The threshold for the air temperature above which ice melt is
            assumed to occur (-1°C the default for monthly mb models). If None
            settings['temp_melt'] is used.
        ys : int, optional
            The start of the climate period where the MB model is valid.
            Defaults to the period with available data.
        ye : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        repeat : bool, default False
            Whether the climate period given by [ys, ye] should be repeated
            indefinitely in a circular way
        check_calib_params : bool, default True
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to ``False`` to suppress
            this check.
        check_climate_data : bool, default True
            If True the climate input data is checked if it is provided in total
            years and that the length matches.
        use_leap_years : bool, default False
            If the calendar should use leap years
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        super(MonthlyTIAvalancheDebrisModel, self).__init__(gdir=gdir,
                                             use_leap_years=use_leap_years)
        self.valid_bounds = [-1e4, 2e4]  # in m
        self.fl_id = fl_id  # which flowline are we the model of?
        self.filename = filename
        self.input_filesuffix = input_filesuffix

        if melt_f is None:  # This prevents class methods
            melt_f = self.calib_params['melt_f']

        if temp_bias is None:
            temp_bias = self.calib_params['temp_bias']

        if prcp_fac is None:
            prcp_fac = self.calib_params['prcp_fac']

        # Check the climate related params to the GlacierDir to make sure
        if check_calib_params:
            mb_calib = self.calib_params['mb_global_params']
            for k, v in mb_calib.items():
                if v != self.gdir.settings[k]:
                    msg = ('You seem to use different mass balance parameters '
                           'than used for the calibration: '
                           f"you use gdir.settings['{k}']={gdir.settings[k]} while "
                           f"it was calibrated with gdir.settings['{k}']={v}. "
                           'Set `check_calib_params=False` to ignore this '
                           'warning.')
                    raise InvalidWorkflowError(msg)
            src = self.calib_params['baseline_climate_source']
            src_calib = gdir.get_climate_info(
                filename=self.filename, input_filesuffix=self.input_filesuffix
            )['baseline_climate_source']
            if src != src_calib:
                msg = (f'You seem to have calibrated with the {src} '
                       f"climate data while this gdir was calibrated with "
                       f"{src_calib}. Set `check_calib_params=False` to "
                       f"ignore this warning.")
                raise InvalidWorkflowError(msg)

        self.melt_f = melt_f
        self.bias = bias

        # Global parameters
        self.temp_all_solid = gdir.settings['temp_all_solid']
        self.temp_all_liq = gdir.settings['temp_all_liq']
        if temp_melt is None:
            self.temp_melt = gdir.settings['temp_melt']
        else:
            gdir.settings['temp_melt'] = temp_melt
            self.temp_melt = temp_melt

        # check if valid prcp_fac is used
        if prcp_fac <= 0:
            raise InvalidParamsError('prcp_fac has to be above zero!')
        self.temp_default_gradient = gdir.settings['temp_default_gradient']

        # Public attrs
        self.hemisphere = gdir.hemisphere
        self.repeat = repeat

        # Add avalanche data
        binned_data_file = gdir.get_filepath(
            "elevation_band_flowline", filesuffix="_fixed_dx"
        )
        binned_data = pd.read_csv(binned_data_file, index_col=0)
        self.ava_prpc_fac = binned_data.snowslide_1m.values

        # Add debris data
        self.debris_ef = binned_data.EF_debris.values

        # Private attrs
        # to allow prcp_fac to be changed after instantiation
        # prescribe the prcp_fac as it is instantiated
        self._prcp_fac = prcp_fac
        # same for temp bias
        self._temp_bias = temp_bias

        # Read climate file
        fpath = gdir.get_filepath(filename, filesuffix=input_filesuffix)
        with ncDataset(fpath, mode='r') as nc:
            time = nc.variables["time"]
            time = cftime.num2date(time[:], time.units, calendar=time.calendar)

            # only use defined years
            years = np.array(list(map(lambda x: x.year, time)))
            pok = slice(None)  # take all is default (optim)
            if ys is not None:
                pok = years >= ys
            if ye is not None:
                try:
                    pok = pok & (years <= ye)
                except TypeError:
                    pok = years <= ye

            self.years = years[pok]
            self.months = np.array(list(map(lambda x: x.month, time)))[pok]
            self.days = np.array(list(map(lambda x: x.day, time)))[pok]

            if check_climate_data:
                # check for full years, this is overwritten for daily
                self._check_for_full_years()

            # Read timeseries and correct it
            self.temp = nc.variables["temp"][pok].astype(np.float64) + self._temp_bias
            self.prcp = nc.variables["prcp"][pok].astype(np.float64) * self._prcp_fac

            grad = self.prcp * 0 + self.temp_default_gradient
            self.grad = grad
            self.ref_hgt = nc.ref_hgt
            self.climate_source = nc.climate_source
            self.ys = self.years[0]
            self.ye = self.years[-1]
            self.ys_float = date_to_floatyear(self.years[0], self.months[0],
                                              self.days[0])
            self.ye_float = date_to_floatyear(self.years[-1], self.months[-1],
                                              self.days[-1])

    def __repr__(self):
        """String Representation of the mass balance model"""
        summary = ['<oggm.MassBalanceModel>']
        summary += ['  Class: ' + self.__class__.__name__]
        summary += ['  Attributes:']
        # Add all scalar attributes
        done = []
        for k in ['hemisphere', 'climate_source', 'melt_f', 'prcp_fac', 'temp_bias', 'bias']:
            done.append(k)
            v = self.__getattribute__(k)
            if k == 'climate_source':
                if v.endswith('.nc'):
                    v = os.path.basename(v)
            nofloat = ['hemisphere', 'climate_source']
            nbform = '    - {}: {}' if k in nofloat else '    - {}: {:.02f}'
            summary += [nbform.format(k, v)]
        for k, v in self.__dict__.items():
            if np.isscalar(v) and not k.startswith('_') and k not in done:
                nbform = '    - {}: {}'
                summary += [nbform.format(k, v)]
        return '\n'.join(summary) + '\n'

    # adds the possibility of changing prcp_fac
    # after instantiation with properly changing the prcp time series
    @property
    def prcp_fac(self):
        """Precipitation factor (default: gdir.settings['prcp_fac'])

        Called factor to make clear that it is a multiplicative factor in
        contrast to the additive temperature bias
        """
        return self._prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, new_prcp_fac):
        # just to check that no invalid prcp_factors are used
        if np.any(np.asarray(new_prcp_fac) <= 0):
            raise InvalidParamsError('prcp_fac has to be above zero!')

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

    @lazy_property
    def calib_params(self):
        if self.fl_id is None:
            return self.gdir.settings
        else:
            fp_fl_settings = self.gdir.get_filepath('settings',
                                                    filesuffix=f'_fl{self.fl_id}')
            if os.path.exists(fp_fl_settings):
                self.gdir.settings_filesuffix = f'_fl{self.fl_id}'
                out = self.gdir.settings
                if self.settings_filesuffix:
                    raise InvalidWorkflowError('settings_filesuffix cannot be '
                                               'used with multiple flowlines')
                return out
            else:
                return self.gdir.settings

    def _check_for_full_years(self):
        # We check for full calendar years
        if self.years[0] != self.years[-1]:
            nr_of_months = (self.years[-1] - self.years[0] + 1) * 12
        else:
            nr_of_months = 12
        len_data_ok = len(self.years) == nr_of_months
        months_ok = (self.months[0] == 1) or (self.months[-1] == 12)
        if not months_ok or not len_data_ok:
            raise InvalidWorkflowError(
                "We now work exclusively with full calendar years. Check "
                "provided climate data! \n Your current selection: "
                f"{self.months[0]:02d}.{self.years[0]:04d} - "
                f"{self.months[-1]:02d}.{self.years[-1]:04d}\n"
                f"Your data has {len(self.years)} timestamps, but we expect "
                f"{nr_of_months}."
            )

    def is_year_valid(self, year: int) -> bool:
        """Check if a year is within the climate period.

        Returns
        -------
        bool
            True if the year is within the climate period.
        """
        return self.ys_float <= year <= self.ye_float

    def validate_year(self, year: int) -> int:
        """Get and validate if a year is outside the data's time range.

        Raises
        ------
        ValueError
            If the year is outside of the data's climate period.
        """
        if self.repeat:
            year = self.ys + (year - self.ys) % (self.ye - self.ys + 1)
        if not self.is_year_valid(year):  # this is overloaded by subclasses
            raise ValueError(
                f'year {year} out of the valid time bounds: '
                f'[{self.ys_float}, {self.ye_float}]'
            )
        return year

    def _get_tempformelt(self, temp):
        tempformelt = temp - self.temp_melt
        clip_min(tempformelt, 0, out=tempformelt)
        return tempformelt

    def _get_prcpsol(self, prcp, temp):
        fac = 1 - (temp - self.temp_all_solid) / (self.temp_all_liq - self.temp_all_solid)
        return prcp * clip_array(fac, 0, 1)

    def _get_climate_for_index(self, heights: ArrayLike, pok: ArrayLike
                               ) -> tuple:
        """Returns climate information at provided heights and time indexes.

        If only one time index is provided also the climate information is
        returned in 1D, otherwise in 2D.

        Parameters
        ----------
        heights : array_like
            the heights of interest in meters
        pok : np.ndarray
            the time indexes of interest

        Returns
        -------
        tuple[np.ndarray]
            Temperature, melt temperatures, total precipitation, and
            solid precipitation for each height pixel.
        """
        # Read already (temperature bias and precipitation factor corrected!)
        itemp = self.temp[pok]
        iprcp = self.prcp[pok]
        igrad = self.grad[pok]

        # For each height pixel:
        heights = np.asarray(heights)  # sometimes heights are passed as lists
        npix = len(heights)

        if np.size(pok) == 1:
            # fast path for 1D data (one time index)
            # Compute temp and tempformelt (temperature above melting threshold)
            temp = itemp + igrad * (heights - self.ref_hgt)
            tempformelt = self._get_tempformelt(temp)

            # Compute solid precipitation from total precipitation
            prcp = np.ones(npix) * iprcp
            prcpsol = self._get_prcpsol(prcp, temp)

            return temp, tempformelt, prcp, prcpsol

        # otherwise we need to handle 2D data
        # Compute temp and tempformelt (temperature above melting threshold)
        grad_temp = np.atleast_2d(igrad).repeat(npix, 0)
        grad_temp *= (heights.repeat(len(pok)).reshape(grad_temp.shape) -
                      self.ref_hgt)
        temp2d = np.atleast_2d(itemp).repeat(npix, 0) + grad_temp
        temp2dformelt = self._get_tempformelt(temp2d)

        # Compute solid precipitation from total precipitation
        prcp = np.atleast_2d(iprcp).repeat(npix, 0)
        prcpsol = self._get_prcpsol(prcp, temp2d)

        return temp2d, temp2dformelt, prcp, prcpsol

    def get_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """
        y, m = floatyear_to_date(year)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) & (self.months == m))[0][0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok,)

        # get tmelt for the entire month
        tmelt *= self.days_in_month(year=year)

        # add precipitation correction factor from avalanching
        ava_prpc_fac = self.ava_prpc_fac
        if len(heights) > len(ava_prpc_fac):
            padsize = len(heights) - len(ava_prpc_fac)
            ava_prpc_fac = np.pad(ava_prpc_fac, (0, padsize), constant_values=1.0)
        prcpsol = prcpsol * ava_prpc_fac

        return t, tmelt, prcp, prcpsol

    def _get_2d_annual_climate(self, heights, year=None):
        # Avoid code duplication with a getter routine
        year = np.floor(year)
        year = self.validate_year(year=year)
        pok = np.where(self.years == year)[0]
        if len(pok) < 1:
            raise ValueError('Year {} not in record'.format(int(year)))

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok)

        return t, tmelt, prcp, prcpsol

    def get_annual_climate(self, heights, year=None):
        """Annual climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Returns
        -------
        tuple
            Mean temperature, and sums of melt temperature,
            precipitation, and solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(
            heights=heights, year=year)
        myr = date_to_floatyear(np.repeat(int(np.floor(year)), 12),
                                np.arange(1, 13))
        days_of_month = [self.days_in_month(year=yr) for yr in myr]
        # get tmelt for the entire month
        tmelt *= days_of_month

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_monthly_mb(self,
                       heights: np.ndarray,
                       year: float = None,
                       add_climate: bool = False,
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self.get_monthly_climate(heights, year=year)

        # length of the month in days already considered in tmelt in get_monthly_climate
        debris_ef = self.debris_ef
        if len(heights) > len(debris_ef):
            padsize = len(heights) - len(debris_ef)
            debris_ef = np.pad(debris_ef, (0, padsize), constant_values=1.0)

        mb_month = np.sum(prcpsol - self.melt_f * tmelt * debris_ef[:, np.newaxis],
                      axis=1) # added debris 

        sec_in_month = self.sec_in_month(year=year)
        mb_month -= (self.bias * sec_in_month / self.sec_in_year(year=year))
        rho = cfg.PARAMS['ice_density']
        if add_climate:
            return mb_month / sec_in_month / rho, t, tmelt, prcp, prcpsol
        return mb_month / sec_in_month / rho

    def get_annual_mb(self, heights, year=None, add_climate=False, **kwargs):
        """Get annual mass balance.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
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

        myr = date_to_floatyear(np.repeat(int(np.floor(year)), 12),
                                np.arange(1, 13))
        days_of_month = [self.days_in_month(year=yr) for yr in myr]
        # get tmelt for the entire month
        tmelt *= days_of_month

        rho = cfg.PARAMS['ice_density']
        
        # Ensure the debris array matches the dynamic flowline length (pad downstream with 1.0)
        current_debris_ef = self.debris_ef
        if len(current_debris_ef) < len(heights):
            padsize = len(heights) - len(current_debris_ef)
            current_debris_ef = np.pad(current_debris_ef, (0, padsize), constant_values=1.0)

        mb_annual = np.sum(prcpsol - self.melt_f * tmelt * current_debris_ef[:, np.newaxis],
                           axis=1)

        mb_annual = ((mb_annual - self.bias) / self.sec_in_year(year=year) /
                     rho)
        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1), ava_added.sum(axis=1),
                    ava_removed.sum(axis=1))
        return mb_annual
    
    def get_annual_acc(self, heights, year=None):
        _, _, _, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)

        acc_annual = np.sum(prcpsol, axis=1)  / SEC_IN_YEAR / rho
        return acc_annual



class DailyTIAvalancheDebrisModel(MonthlyTIModel):
    """Daily temperature index model."""

    def __init__(
        self,
        gdir,
        filename: str = 'climate_historical_daily',
        input_filesuffix: str = '',
        settings_filesuffix: str = '_with_avadeb',
        fl_id: int = None,
        melt_f: float = None,
        temp_bias: float = None,
        prcp_fac: float = None,
        bias: float = 0.0,
        temp_melt: float = 0.0,
        ys: int = None,
        ye: int = None,
        repeat: bool = False,
        check_calib_params: bool = True,
        check_climate_data: bool = True,
        use_leap_years: bool = True,
    ):
        """Inherits from MonthlyTIModel.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        filename : str, default 'climate_historical_daily'
            Set to a different BASENAME if you want to use alternative
            climate data.
        input_filesuffix : str, optional
            Append a suffix to the climate input filename (useful for
            GCM runs).
        settings_filesuffix : str, optional
            Append a suffix to the settings file (useful for sensitivity
            runs).
        fl_id : int, optional
            If this flowline has been calibrated alone and has specific
            model parameters.
        melt_f : float, optional
            The value of the melt factor you want to use, here the unit
            is kg m-2 day-1 K-1. Defaults to the calibrated value.
        temp_bias : float, optional
            The value of the temperature bias. Defaults to the
            calibrated value.
        prcp_fac : float, optional
            The value of the precipitation factor. Defaults to the
            calibrated value.
        bias : float, default 0.0
            The value of the calibration bias [mm we yr-1]. Defaults to
            the calibrated value. Note that this bias is *subtracted*
            from the computed MB. Indeed:
            BIAS = MODEL_MB - REFERENCE_MB.
        temp_melt : float, default 0.0
            The threshold for the air temperature above which ice melt is
            assumed to occur (0°C the default for daily mb models).
        ys : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        ye : int, optional
            The end of the climate period where the MB model is valid.
            Defaults to the period with available data.
        repeat : bool, default False
            Whether the climate period given by [ys, ye] should be
            repeated indefinitely in a circular way.
        check_calib_params : bool, default True
            OGGM will try hard not to use wrongly calibrated parameters
            by checking the global parameters used during calibration
            and the ones you are using at run time. If they don't
            match, it will raise an error. Set to ``False`` to suppress
            this check.
        check_climate_data : bool, default True
            If True, check the climate input data is provided in total
            years and that the length matches.
        use_leap_years : bool, default True
            If the calendar should use leap years.
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        super(DailyTIAvalancheDebrisModel, self).__init__(
            gdir=gdir,
            filename=filename,
            input_filesuffix=input_filesuffix,
            settings_filesuffix=settings_filesuffix,
            fl_id=fl_id,
            melt_f=melt_f,
            temp_bias=temp_bias,
            prcp_fac=prcp_fac,
            bias=bias,
            temp_melt=temp_melt,
            ys=ys,
            ye=ye,
            repeat=repeat,
            check_calib_params=check_calib_params,
            check_climate_data=check_climate_data,
            use_leap_years=use_leap_years,
        )

        # Add avalanche data
        binned_data_file = gdir.get_filepath(
            "elevation_band_flowline", filesuffix="_fixed_dx"
        )
        binned_data = pd.read_csv(binned_data_file, index_col=0)
        self.ava_prpc_fac = binned_data.snowslide_1m.values

        # Add debris data
        self.debris_ef = binned_data.EF_debris.values

    def _check_for_full_years(self):
        # We check for full calendar years
        nr_of_days = 0
        for yr in np.arange(self.years[0], self.years[-1] + 1):
            nr_of_days += self.days_in_year(yr)
        len_data_ok = len(self.years) == nr_of_days
        months_ok = (self.months[0] == 1) and (self.months[-1] == 12)
        days_ok = (self.days[0] == 1) and (self.days[-1] == 31)
        if not months_ok or not days_ok or not len_data_ok:
            raise InvalidWorkflowError(
                "We now work exclusively with full calendar years (01.01. - "
                "31.12.). Check provided climate data!\nYour current selection: "
                f"{self.days[0]:02d}.{self.months[0]:02d}.{self.years[0]:04d} - "
                f"{self.days[-1]:02d}.{self.months[-1]:02d}.{self.years[-1]:04d}"
                f"\nYour data has {len(self.years)} timestamps, but we expect "
                f"{nr_of_days}."
            )

    def get_annual_climate(self, heights, year=None):
        t, tmelt, prcp, prcpsol = self._get_2d_annual_climate(heights, year)

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def _get_2d_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        y, m = floatyear_to_date(year)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) & (self.months == m))[0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok)

        return t, tmelt, prcp, prcpsol

    def get_monthly_climate(
        self, heights: np.ndarray, year: float = None
    ) -> tuple:
        """Monthly climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """

        t, tmelt, prcp, prcpsol = self._get_2d_monthly_climate(heights, year)

        # add precipitation correction factor from avalanching
        prcpsol = prcpsol * self.ava_prpc_fac

        return (t.mean(axis=1), tmelt.sum(axis=1),
                prcp.sum(axis=1), prcpsol.sum(axis=1))

    def get_daily_climate(self,
                          heights: np.ndarray,
                          year: float = None,
                          ) -> tuple:
        """Daily climate information at given heights.

        Note that prcp is corrected with the precipitation factor and that
        all other model biases (temp and prcp) are applied.

        Parameters
        ----------
        heights : np.ndarray[np.float64]
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.

        Returns
        -------
        tuple[np.ndarray]
            Temperatures, melt temperatures, total precipitation, and
            solid precipitation.
        """
        y, m, d = floatyear_to_date(year, return_day=True)
        y = self.validate_year(year=y)
        pok = np.where((self.years == y) &
                       (self.months == m) &
                       (self.days == d))[0][0]

        t, tmelt, prcp, prcpsol = self._get_climate_for_index(
            heights=heights, pok=pok, )
        
        # add precipitation correction factor from avalanching
        ava_prpc_fac = self.ava_prpc_fac
        if len(heights) > len(ava_prpc_fac):
            padsize = len(heights) - len(ava_prpc_fac)
            ava_prpc_fac = np.pad(ava_prpc_fac, (0, padsize), constant_values=1.0)
        prcpsol = prcpsol * ava_prpc_fac

        return t, tmelt, prcp, prcpsol

    def get_daily_mb(self,
                     heights: np.ndarray,
                     year: int = None,
                     add_climate: bool = False,
                     **kwargs,
                     ) -> np.float64 or tuple:
        """Get daily mass balance.

        Accounts for leap years by default.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : int, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Daily mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self.get_daily_climate(heights, year=year)

        debris_ef = self.debris_ef
        if len(heights) > len(debris_ef):
            padsize = len(heights) - len(debris_ef)
            debris_ef = np.pad(debris_ef, (0, padsize), constant_values=1.0)

        mb_daily = prcpsol - self.melt_f * tmelt * debris_ef[:, np.newaxis] # added debris

        mb_daily -= (self.bias * SEC_IN_DAY / self.sec_in_year(year=year))
        rho = cfg.PARAMS['ice_density']

        if add_climate:
            return (mb_daily / SEC_IN_DAY / rho,
                    t, tmelt, prcp, prcpsol)
        return mb_daily / SEC_IN_DAY / rho

    def get_monthly_mb(self,
                       heights: np.ndarray,
                       year: float = None,
                       add_climate: bool = False,
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly mass balance.

        Parameters
        ----------
        heights : np.ndarray
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to subclass implementations of this
            method.


        Returns
        -------
        np.float64 or tuple[np.ndarray]
            Monthly mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
        t, tmelt, prcp, prcpsol = self._get_2d_monthly_climate(heights, year)

        mb_month = np.sum(prcpsol - self.melt_f * tmelt * self.debris_ef[:, np.newaxis],
                          axis=1)
        sec_in_month = self.sec_in_month(year=year)
        mb_month -= (self.bias * sec_in_month / self.sec_in_year(year=year))
        rho = cfg.PARAMS['ice_density']

        if add_climate:
            return (mb_month / sec_in_month / rho, t.mean(axis=1),
                    tmelt.sum(axis=1), prcp.sum(axis=1), prcpsol.sum(axis=1))

        return mb_month / sec_in_month / rho

    def get_annual_mb(self,
                      heights: np.ndarray,
                      year: float = None,
                      add_climate: bool = False,
                      **kwargs,
                      ) -> np.float64 or tuple:
        """Get annual mass balance.

        This is equivalent to taking the sum of ``get_daily_mb``.

        Parameters
        ----------
        heights : array_like
            Heights in m.
        year : float, optional
            The year (in the "floating year" convention). Default None.
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        **kwargs
            Extra arguments passed to ``get_2d_temperature``.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual mass balance in metres of ice per second. If
            ``add_climate`` is True, also returns mean temperature and
            the sums of melt temperature, total precipitation, and
            solid precipitation.
        """
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
        rho = cfg.PARAMS['ice_density']

        # Ensure the debris array matches the dynamic flowline length (pad downstream with 1.0)
        current_debris_ef = self.debris_ef
        if len(current_debris_ef) < len(heights):
            padsize = len(heights) - len(current_debris_ef)
            current_debris_ef = np.pad(current_debris_ef, (0, padsize), constant_values=1.0)

        mb_annual = np.sum(prcpsol - self.melt_f * tmelt * current_debris_ef[:, np.newaxis],
                           axis=1)
        mb_annual = ((mb_annual - self.bias) / self.sec_in_year(year=year) /
                     rho)

        if add_climate:
            return (mb_annual, t.mean(axis=1), tmelt.sum(axis=1),
                    prcp.sum(axis=1), prcpsol.sum(axis=1), ava_added.sum(axis=1),
                    ava_removed.sum(axis=1))
        return mb_annual
    
    def get_annual_acc(self, heights, year=None):
        _, _, _, prcpsol = self._get_2d_annual_climate(heights, year)

        # We add avalanches!!!
        if prcpsol.shape[0] < self.ava_prpc_fac.shape[0]:
            raise InvalidWorkflowError(
                "Avalanche model needs to be " "called on the entire glacier " "domain"
            )
        padsize = prcpsol.shape[0] - self.ava_prpc_fac.shape[0]
        ava_prpc_fac = np.pad(self.ava_prpc_fac, (0, padsize), constant_values=(1, 1))

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)
        rho = cfg.PARAMS['ice_density']
        
        acc_annual = np.sum(prcpsol, axis=1)  / SEC_IN_YEAR / rho
        return acc_annual



class SfcTypeTIAvalancheDebrisModel(MassBalanceModel):
    """Temperature-Index model with surface type distinction using a bucket
    system.

    Adapted from OGGM/massbalance-sandbox and explained in Schuster et al. 2023.
    """

    def __init__(
        self,
        gdir,
        settings_filesuffix: str = "_with_ava_deb",
        use_leap_years: bool = True,
        mb_model_class=DailyTIAvalancheDebrisModel,
        climate_resolution: str = "annual",
        aging_frequency: str = "annual",
        melt_f_ratio: float = 0.5,
        melt_f_change: str = "neg_exp",
        tau_e: float = 1.0,
        ys: int = 2000,
        spinup_years: int = 6,
        save_spinup_mbs: bool = False,
        spinup_buckets: np.ndarray = None,
        fl = None,
        use_main_fl_from: str = 'inversion_flowlines',
        hbins: ArrayLike = None,
        store_buckets: bool or str = False,
        store_buckets_dates: ArrayLike = None,
        use_previous_mbs: bool = False,
        store_snowline: bool = False,
        store_snowline_start_month: str = 'Oct',
        **kwargs,
    ):
        """Surface type temperature index model.

        Parameters
        ----------
        gdir : GlacierDirectory
            The glacier directory.
        settings_filesuffix : str, optional
            append a suffix to the settings file (useful for
            sensitivity runs).
        use_leap_years : bool, default True
            If the calendar should use leap years.
        mb_model_class : MassBalanceModel, default ``DailyTIModel``
            The MassBalanceModel to use together with surface type tracking.
            Options are DailyTIModel and MonthlyTIModel.
        climate_resolution : str, default 'annual'
            Temporal mass balance resolution. Options are 'annaul', 'monthly'
            and 'daily', but the mb_model_class need to be able to provide this
            resolution.
        aging_frequency : str, default "annual"
            Frequency at which the buckets are moved to the next one (= getting
            older). Options are "annual" or "monthly".
            If annual, the model uses one snow and (spinup_years-1) firn buckets.
            If monthly, the snow ages over the number of months * spinup_years.
        melt_f_ratio : float, default 0.5
            Ratio of snow melt factor to ice melt factor.
            Between 0 and 1, where 1 is no surface type distinction.
            Default 0.5 to match GloGEM.
        melt_f_change : str, default "neg_exp"
            How the snow melt factor changes relative to the ice melt
            factor, either "linear" or "neg_exp" (see `tau_e` for the equation).
        tau_e : float, default 1
            Only used if melt_f_change is 'neg_exp'. It describes how fast the
            snow melt factor approximates to the ice melt factor via
            melt_f=melt_f_ice+(melt_f_snow-melt_f_ice)*np.exp(-time_yr/tau_e_fold_yr)
            Must be larger than zero to prevent ``melt_f`` being set to NaN in
            the first bucket.
        ys : int, default 2000
            The initial year from where we want to get mb values. This means the
            bucket system is spun up so that at ys for the first time ice
            can form. For spinup the years ys - spinup_years up to ys are used.
        spinup_years : int, default 6
            Number of spinup years. This defines the number of buckets we use
            (see aging_frequency for explanation). The minimum allowed value is
            1 (corresponds to only having a snow bucket for
            aging_frequency='annual'). The spinup ensures every bucket has the
            opportunity to be filled and ice can form at ys.
        save_spinup_mbs : bool, default False
            If we want to safe the climatic mb and ice mb during the spinup
            period. This is mainly useful for testing.
        spinup_buckets : np.ndarray, default None
            Instead of conducting a spinup of the buckets you can provide the
            initial buckets. It is assumed that these buckets are valid at the
            time ys.
        fl: Flowline, default None
            The flowline for defining the grid-points. If None, the flowline
            defined in `use_main_fl_from` is used. During all consecutive calls
            it is always checked the length of provided heights equals the
            length of the grid-points of the flowline.
        use_main_fl_from: str, default 'inversion_flowlines'
            If the main flowline of 'inversion_flowlines' or 'model_flowlines'
            should be used. If fl is provided this is ignored.
        hbins: ArrayLike, default None
            Height bins for classifying surface types. Only needed for
            ``ConstantMBModel``. Currently untested!
        store_buckets: bool or str, default False
            If buckets should be stored along the way. Options are 'annual',
            'monthly', 'daily' or False.
        store_buckets_dates: ArrayLike, default []
            In addition to the frequency of when the buckets should be stored
            defined in `store_buckets` you can also provide a few selected dates
            where the buckets should be stored.
        use_previous_mbs: bool, default False
            If True, and you call get_annual_mb, get_monthly_mb or get_daily_mb
            it is checked if the desired year was already computed and return
            this values. In this case the provided heights will be ignored. This
            can be needed for some tasks (e.g. run_with_hydro), but the user
            should be aware that this behaviour is different than other
            mb_models work, because with surface tracking we have included a
            memory of the past. If False and you try to revisit a past year and
            error is raised.
        store_snowline: bool, default False
            Experimental: save the snowline in the climate_resolution. For this
            the lowest elevation where the bucket is not zero is set as the
            snowline. Older buckets are considered starting from
            store_snowline_start_month.
        store_snowline_start_month: str, default 'Oct'
            Experimental: the start month of buckets used for the derivation of
            the snowline. E.g. with the default 'Oct', if you want to derive the
            snowline at Mar, all buckets starting from the previous Oct are
            considered.
        **kwargs:
            keyword arguments to pass to the mb_model_class
        """

        self.settings_filesuffix = settings_filesuffix
        gdir.settings_filesuffix = settings_filesuffix

        super(SfcTypeTIAvalancheDebrisModel, self).__init__(gdir=gdir,
                                             use_leap_years=use_leap_years)

        # the mass balance model we use to get the climate data, as well as
        # melt_f, prcp_fac and temp_bias
        if mb_model_class not in [MonthlyTIAvalancheDebrisModel, DailyTIAvalancheDebrisModel]:
            raise NotImplementedError(f"mb_model_class: {mb_model_class}")
        self.mbmod = mb_model_class(gdir=gdir,
                                    settings_filesuffix=settings_filesuffix,
                                    **kwargs)
        self.filename = self.mbmod.filename
        self.input_filesuffix = self.mbmod.input_filesuffix
        self.hemisphere = self.mbmod.hemisphere
        self.bias = self.mbmod.bias
        self.ys = self.mbmod.ys
        self.ye = self.mbmod.ye
        self.debris_ef = self.mbmod.debris_ef

        # check compatibility of aging_frequency and climate_resolution: aging
        # can not happen at higher temporal resolution than the climate steps
        if aging_frequency not in ["annual", "monthly"]:
            raise NotImplementedError(f"aging_frequency. {aging_frequency}")
        climate_and_aging_compatible = True
        if climate_resolution == 'annual':
            if aging_frequency != 'annual':
                climate_and_aging_compatible = False
        elif climate_resolution in ['monthly', 'daily']:
            # currently the fastest aging supported is monthly
            pass
        else:
            raise NotImplementedError(
                f"climate_resolution: {climate_resolution}")
        if not climate_and_aging_compatible:
            raise InvalidParamsError(
                f"The selected climate_resolution '{climate_resolution}' is "
                f"not compatible with an aging_frequency '{aging_frequency}'. "
                "The climate resolution must be the same or shorter than "
                "aging (e.g. you can not age monthly while the climate is "
                "only applied annual).")
        self.aging_frequency = aging_frequency
        self.climate_resolution = climate_resolution

        # stuff related to spinup, only used if spinup_buckets is None
        if spinup_years < 1:
            raise InvalidParamsError("spinup_years needed to be at least 1, you "
                                     f"provided {spinup_years}.")
        self.spinup_years = spinup_years
        self.save_spinup_mbs = save_spinup_mbs
        self.spinup_buckets = spinup_buckets
        if isinstance(self.spinup_buckets, pd.core.frame.DataFrame):
            self.spinup_buckets = self.spinup_buckets.values
        self.ys = ys
        if self.spinup_buckets is None:
            # check if climate data for spinup is available
            spinup_start = self.ys - self.spinup_years
            try:
                self.mbmod.validate_year(spinup_start)
                self.mbmod.validate_year(self.ys)
            except ValueError as e:
                raise ValueError(
                    "Climate data for spinup not available. We need data for "
                    f"the period {spinup_start} (ys - spinup_years) to "
                    f"{self.ys}, but we get the following error: {e}")

        # defining the number of grid points and the spinup heights, either with
        # fl or hbins
        self.hbins = hbins  # TODO: need to test with constant_mb (maybe hbins[::-1])
        if fl is not None:
            self.fl = fl
        else:
            if use_main_fl_from == 'inversion_flowlines':
                self.fl = self.gdir.read_pickle("inversion_flowlines")[-1]
            elif use_main_fl_from == 'model_flowlines':
                self.fl = self.gdir.read_pickle("model_flowlines")[-1]
            elif self.hbins is not None:
                # ok we are working with height bins here, don't need the fl
                self.fl = None
            else:
                raise InvalidParamsError("We need a flowline or height bins "
                                         "(hbins) for defining the number of "
                                         "buckets we want to compute!")

        # create labels for the grid_points and define heights used during spinup
        if self.fl is not None:
            # for the flowline we use the distance along the flowline
            self.buckets_grid_point_label = self.fl.dx_meter * np.arange(self.fl.nx)
            self.spinup_heights = self.fl.surface_h
        else:
            # for hbins we just use numbers
            self.buckets_grid_point_label = np.arange(len(self.hbins))
            self.spinup_heights = self.hbins

        # here we set the names of the buckets, this depends on the
        # aging_frequency and spinup_years
        if self.aging_frequency == "annual":
            firn_buckets = [f"firn_{i}" for i in range(1, self.spinup_years, 1)]
        elif self.aging_frequency == "monthly":
            firn_buckets = [f"firn_{i}" for i in
                            range(1, 12 * self.spinup_years, 1)]
        else:
            raise InvalidParamsError("aging_frequency must be 'annual' or "
                                     "'monthly'")

        # we also add a snow and an ice bucket
        self.buckets = ["snow"] + firn_buckets + ["ice"]

        # set a template for an empty bucket and mb containers
        self._empty_mb_buckets_np = np.zeros((len(self.buckets_grid_point_label),
                                              len(self.buckets)))

        # stuff related to varying melt_f for each bucket
        self.tau_e = tau_e
        self.melt_f_ratio = melt_f_ratio
        self.melt_f_change = melt_f_change
        self.melt_f_buckets = None
        self.set_melt_f_buckets()

        # define if some additonal outputs should be saved along the way
        if ((store_buckets == 'monthly' and
             self.climate_resolution == 'annual') or
                (store_buckets == 'daily' and
                 self.climate_resolution != 'daily')):
            raise InvalidParamsError(
                "It is not possible to save the buckets in a higher resolution "
                "than the applied climate! Your selection: store_buckets = "
                f"{store_buckets}, climate_resolution = "
                f"{self.climate_resolution}")
        self.store_buckets = store_buckets
        if store_buckets_dates is not None:
            # if specific dates are provided this overrules store_buckets
            self.store_buckets = True
        self.store_buckets_dates = store_buckets_dates

        # if the user wants to return previously calculated mb values
        self.use_previous_mbs = use_previous_mbs

        # snowline stuff
        self.store_snowline = store_snowline
        if self.store_snowline:
            # how many buckets we need to add for the snowline buckets
            self._snowline_start_month = {
                "Jan": 11, "Feb": 10, "Mar": 9, "Apr": 8, "May": 7, "Jun": 6,
                "Jul": 5, "Aug": 4, "Sep": 3, "Oct": 2, "Nov": 1, "Dec": 0,
            }[store_snowline_start_month]
            self._snowline = []
            self._snowline_year = []
            # this are the height values for the special cases fully snow
            # covered or fully snow free
            self.snowline_inf_values = {}

        # Initialise buckets and conduct a potential spinup
        self._init_buckets()

    def _init_buckets(self):
        # reset some containers for a fresh start
        self.mb_buckets_np = self._empty_mb_buckets_np.copy()  # kg m-2
        # define length of needed timesteps
        first_year = self.ys
        if self.save_spinup_mbs:
            first_year -= self.spinup_years
        if self.climate_resolution == 'annual':
            self.nr_timesteps = len(range(first_year, self.ye + 1))
        elif self.climate_resolution == 'monthly':
            self.nr_timesteps = len(float_years_timeseries(
                y0=first_year, y1=self.ye, include_last_year=True, ))
        elif self.climate_resolution == 'daily':
            self.nr_timesteps = len(float_years_timeseries(
                y0=first_year, y1=self.ye, include_last_year=True,
                daily=True))
        else:
            raise NotImplementedError(
                f"climate_resolution {self.climate_resolution}")

        output_shape = (self.nr_timesteps,  # total nr of timesteps
                        len(self.buckets_grid_point_label))  # nr of grid points
        self._climatic_mb = np.empty(output_shape)  # kg m-2
        self._ice_mb = np.empty(output_shape)  # kg m-2
        self._mb_heights = np.empty(output_shape)  # m
        self._year_to_index = {}  # saving the array positions of years
        self._current_index = 0  # keep track of last added position

        if self.store_buckets:
            # the mb_buckets are stored in a dict, with key corresponding to the
            # date and the buckets saved as pandas dataframes
            self.mb_buckets_stored = {}

        # now define the initial buckets
        if self.spinup_buckets is not None:
            # ok user provided the initial buckets, check shape and use
            provided_shape = self.spinup_buckets.shape
            needed_shape = self._empty_mb_buckets_np.shape
            if (provided_shape[0] != needed_shape[0] or
                    provided_shape[1] != needed_shape[1] - 1):  # exclude ice
                raise InvalidParamsError(
                    "The provided spinup_buckets have the shape "
                    f"{provided_shape}, but we expect the shape "
                    f"({needed_shape[0]}, {needed_shape[1] - 1}).")

            self.mb_buckets_np = self._empty_mb_buckets_np.copy()
            self.mb_buckets_np[:, :-1] = self.spinup_buckets.copy()
            self.mb_buckets_year = self.ys

        else:
            # we need to make a spinup
            # define the spinup timesteps depending on the climate resolution
            spinup_start_year = self.ys - self.spinup_years
            if self.climate_resolution == 'annual':
                spinup_steps = range(spinup_start_year, self.ys)
            elif self.climate_resolution == 'monthly':
                spinup_steps = float_years_timeseries(
                    y0=spinup_start_year, y1=self.ys)[:-1]
            elif self.climate_resolution == 'daily':
                spinup_steps = float_years_timeseries(
                    y0=spinup_start_year, y1=self.ys, daily=True)[:-1]
            else:
                raise NotImplementedError(
                    f"climate_resolution {self.climate_resolution}")

            # do the actual spinup
            for yr in spinup_steps:
                self._apply_climate_step_and_aging_to_buckets(
                    heights=self.spinup_heights, year=yr,
                    save_mbs=self.save_spinup_mbs)

            assert self.mb_buckets_year == self.ys

    @property
    def mb_buckets(self):
        # returns current snow and firn buckets, but not the ice bucket,
        # for ice look at ice_mb
        return pd.DataFrame(self.mb_buckets_np[:, :-1],
                            index=self.buckets_grid_point_label,
                            columns=self.buckets[:-1],)

    @property
    def climatic_mb(self):
        pd_dict = {}
        for year in self._year_to_index:
            pd_dict[year] = self._climatic_mb[[self._year_to_index[year]]][0]
        return pd.DataFrame(pd_dict,
                            index=self.buckets_grid_point_label)

    @property
    def ice_mb(self):
        pd_dict = {}
        for year in self._year_to_index:
            pd_dict[year] = self._ice_mb[[self._year_to_index[year]]][0]
        return pd.DataFrame(pd_dict,
                            index=self.buckets_grid_point_label)

    @property
    def mb_heights(self):
        pd_dict = {}
        for year in self._year_to_index:
            pd_dict[year] = self._mb_heights[[self._year_to_index[year]]][0]
        return pd.DataFrame(pd_dict,
                            index=self.buckets_grid_point_label)

    @property
    def snowline(self):
        return np.array(self._snowline)

    @property
    def snowline_year(self):
        return np.array(self._snowline_year)

    @property
    def melt_f(self):
        return self.mbmod.melt_f

    @melt_f.setter
    def melt_f(self, value):
        """Set new melt_f and reset the buckets."""
        self.mbmod.melt_f = value
        self.set_melt_f_buckets()
        # Recompute buckets
        self._init_buckets()

    @property
    def prcp_fac(self):
        """Precipitation factor"""
        return self.mbmod.prcp_fac

    @prcp_fac.setter
    def prcp_fac(self, value):
        """Set new precipitation factor and reset buckets."""
        self.mbmod.prcp_fac = value
        # Recompute buckets
        self._init_buckets()

    @property
    def temp_bias(self):
        return self.mbmod.temp_bias

    @temp_bias.setter
    def temp_bias(self, value):
        """Set new temperature bias and reset buckets."""
        self.mbmod.temp_bias = value
        # Recompute buckets
        self._init_buckets()

    def set_melt_f_buckets(self):
        """Set the melt factor for each bucket."""
        if self.melt_f_change == "linear":
            self.melt_f_buckets = dict(
                zip(self.buckets,
                    np.linspace(self.melt_f * self.melt_f_ratio, self.melt_f,
                                len(self.buckets),),
                    )
            )
        elif self.melt_f_change == "neg_exp":
            if self.tau_e <= 0:
                raise InvalidParamsError("`tau_e` must be above zero for"
                                         "`melt_f_change` = 'neg_exp'.")
            buckets_linspace = np.linspace(0, self.spinup_years,
                                           len(self.buckets))
            melt_f_snow = self.melt_f_ratio * self.melt_f
            self.melt_f_buckets = dict(
                zip(self.buckets,
                    self.melt_f + (melt_f_snow - self.melt_f) *
                    np.exp(-buckets_linspace / self.tau_e),
                    )
            )
        else:
            raise NotImplementedError(f"melt_f_change: {self.melt_f_change}")

    def _apply_climate_step_and_aging_to_buckets(
        self,
        heights: ArrayLike,
        year: float = None,
        save_mbs: bool = True,
    ):
        """ This is the central method where the magic is happening.

        It gets the climate input data for the provided year, adds fresh snow to
        the snow bucket, and melts the buckets in the correct order. At the end
        it checks if aging is needed, which moves everything into the next older
        bucket.

        Parameters
        ----------

        heights: array_like
           heights used to get the climate input data, need to be the same
           length as the number of defined grid points and in the same order.
           The number of grid points was defined either through fl or hbins at
           initialisation.
        year: float
           year in calendar float year
        save_mbs: bool, default True
            If True, the climatic_mb and the ice_mb are saved. This is set to
            False during spinup

        """

        # Need to provide heights for the same grid points as the bucket was
        # initialized. They also need to be in the same order, but this we can
        # not check
        if len(heights) != len(self.buckets_grid_point_label):
            raise InvalidParamsError(f'Number of heights ({len(heights)}) does '
                                     'not match number of grid points provided '
                                     'on initialization '
                                     f'({len(self.buckets_grid_point_label)}).')

        # get the current buckets, first is snow, last is ice, inbetween firn
        mb_buckets_np = self.mb_buckets_np

        # get the climate data in the defined resolution
        if self.climate_resolution == "annual":
            t, tmelt, prcp, prcpsol = self.mbmod.get_annual_climate(heights, year)
        elif self.climate_resolution == "monthly":
            t, tmelt, prcp, prcpsol = self.mbmod.get_monthly_climate(heights, year)
        elif self.climate_resolution == "daily":
            t, tmelt, prcp, prcpsol = self.mbmod.get_daily_climate(heights, year)
        else:
            raise NotImplementedError(
                f"climate_resolution: {self.climate_resolution}")

        # now we use tmelt for melting the buckets in order, the following
        # algorithm is optimized for np arrays smaller to shape (1000, 72),
        # which corresponds to 1000 flowline grid points and 72 buckets (= 6
        # years with monthly aging). We do this only for the snow and firn
        # buckets and deal with ice afterwards. For checking the code I add the
        # shape of variables as comments.

        # here we will save the results, use view to modify in-place
        snow_buckets_new = mb_buckets_np[:, :-1]
        nr_grid_points, nr_buckets = snow_buckets_new.shape
        # we need the sum of the old buckets later for calculating delta kg m-2
        snow_buckets_old_sum = snow_buckets_new.sum(axis=1)
        # melt_f per bucket without ice, (1, nr_buckets)
        melt_f_buckets = np.asarray(list(self.melt_f_buckets.values())[:-1],
                                    dtype=float)[None, :]
        # add one axis to tmelt for correct shape, (nr_grid_points, 1)
        tmelt = tmelt[:, None]

        # all solid precip. goes into fresh snow bucket before we start melting
        snow_buckets_new[:, 0] += prcpsol

        # now calculate cumulative tfm needed to melt each bucket and subtract
        # available tmelt, finally we convert back to mass in each bucket
        # (nr_grid_points, nr_buckets)
        buckets_kg_m2_cumsum_left = (((snow_buckets_new / melt_f_buckets
                                       ).cumsum(axis=1) - tmelt) *
                                     melt_f_buckets)

        # when the cumsum is negative this bucket and all above have melted
        # completely, (nr_grid_points, nr_buckets)
        melted_buckets = buckets_kg_m2_cumsum_left < 0

        # number of fully melted buckets per grid point, (nr_grid_points,)
        nr_melted_buckets = melted_buckets.sum(axis=1)

        # set melted buckets to zero
        snow_buckets_new[melted_buckets] = 0.0

        # set buckets which have only partially melted
        some_left_buckets = np.where(nr_melted_buckets < nr_buckets)[0]
        if some_left_buckets.size:
            partly_melted_bucket = nr_melted_buckets[some_left_buckets]
            # Set the partially melted bucket to the leftover mass
            snow_buckets_new[some_left_buckets, partly_melted_bucket] = (
                buckets_kg_m2_cumsum_left[some_left_buckets, partly_melted_bucket])

        # Now deal with ice, ice melts where some tmelt is still left after
        # melting all buckets
        all_melted_grid_points = (nr_melted_buckets == nr_buckets)
        if np.any(all_melted_grid_points):
            remaining_tfm = -(buckets_kg_m2_cumsum_left[all_melted_grid_points, -1] /
                              melt_f_buckets[:, -1])

            debris_ef = self.debris_ef
            if len(all_melted_grid_points) > len(debris_ef):
                padsize = len(all_melted_grid_points) - len(debris_ef)
                debris_ef = np.pad(debris_ef, (0, padsize), constant_values=1.0)

            ice_melt_kg_m2 = (remaining_tfm * self.melt_f_buckets['ice'] * debris_ef[all_melted_grid_points])
            
            # we use -= here because there could be some newly formed ice
            # already in the ice bucket after aging
            self.mb_buckets_np[all_melted_grid_points, -1] -= ice_melt_kg_m2
        else:
            ice_melt_kg_m2 = None

        if save_mbs:
            # Calculate the climatic mb as delta of buckets and minus potential
            # ice melt, all in kg m-2
            delta_kg_m2 = (snow_buckets_new.sum(axis=1)  # sum of buckets at the end
                           - snow_buckets_old_sum)  # sum of buckets at the start
            if ice_melt_kg_m2 is not None:
                delta_kg_m2[all_melted_grid_points] -= ice_melt_kg_m2  # ice melt

            # save the climatic mb of this timestep
            self._climatic_mb[self._current_index] = delta_kg_m2

        # save the mb of ice, this includes potential ice gain from aging after
        # the call of _bucket_aging and melt where all snow/firn buckets are
        # empty
        if save_mbs:
            self._ice_mb[self._current_index] = self.mb_buckets_np[:, -1].copy()
        # we empty the ice bucket after saving to avoid any double counting
        self.mb_buckets_np[:, -1] = 0.0

        if save_mbs:
            self._mb_heights[self._current_index] = heights

        # at the end we save the year of current index and increase it
        if save_mbs:
            self._year_to_index[year] = self._current_index
            self._current_index += 1

        # update the current year of the buckets, this is set one timestep later
        # to the currently applied climate step (e.g. after applying the climate
        # of January the bucket refers to the first of February)
        if self.climate_resolution == 'annual':
            buckets_yr = int(year) + 1
            buckets_month = 1
            buckets_day = 1
            self.mb_buckets_year = buckets_yr
        elif self.climate_resolution == 'monthly':
            buckets_yr, buckets_month = floatyear_to_date(float(year))
            buckets_day = 1
            # special case when a new year starts
            if buckets_month == 12:
                buckets_yr += 1
                buckets_month = 1
            else:
                buckets_month += 1
            self.mb_buckets_year = date_to_floatyear(buckets_yr, buckets_month)
        elif self.climate_resolution == 'daily':
            buckets_yr, buckets_month, buckets_day = floatyear_to_date(
                float(year), return_day=True)
            # date_to_floatyear can deal with monthly and yearly overflows
            # (e.g. 32.01. == 01.02., and 32.12.2000 == 01.01.2001)
            self.mb_buckets_year = date_to_floatyear(
                buckets_yr, buckets_month, buckets_day + 1)
            # finally we get the values without the overflow for aging below
            buckets_yr, buckets_month, buckets_day = floatyear_to_date(
                self.mb_buckets_year, return_day=True
            )
        else:
            raise NotImplementedError(
                f"climate_resolution: {self.climate_resolution} ")

        # Finally the bucket aging, potentially new formed ice is considered in
        # the next climate step
        if self.aging_frequency == 'annual':
            if self.climate_resolution == 'annual':
                # we need to age each step
                self._bucket_aging()
            elif self.climate_resolution == 'monthly':
                # we age if the bucket month is January (bucket year is always
                # one timestep ahead of current climate year, see above)
                if buckets_month == 1:
                    self._bucket_aging()
            elif self.climate_resolution == 'daily':
                if buckets_month == 1 and buckets_day == 1:
                    self._bucket_aging()
            else:
                raise NotImplementedError(
                    f"climate_resolution: {self.climate_resolution} ")
        elif self.aging_frequency == 'monthly':
            if self.climate_resolution == 'monthly':
                # we need to age each step
                self._bucket_aging()
            elif self.climate_resolution == 'daily':
                if buckets_day == 1:
                    self._bucket_aging()
            else:
                # it is already checked at initialisation that
                # climate_resolution is <= aging_frequency, but if we have
                # overseen something, this should raise
                raise NotImplementedError(
                    f"climate_resolution: {self.climate_resolution} ")
        else:
            raise NotImplementedError(
                f"aging_frequency: {self.aging_frequency} ")

        # check if the resulting bucket should be stored for later
        if self.store_buckets:
            store_current_bucket = False
            if self.store_buckets_dates is not None:
                if self.mb_buckets_year in self.store_buckets_dates:
                    store_current_bucket = True
            elif self.store_buckets == 'annual':
                if buckets_month == 1 and buckets_day == 1:
                    store_current_bucket = True
            elif self.store_buckets == 'monthly':
                if buckets_day == 1:
                    store_current_bucket = True
            elif self.store_buckets == 'daily':
                store_current_bucket = True
            else:
                raise NotImplementedError(f"store_buckets: {self.store_buckets}")

            if store_current_bucket:
                self.mb_buckets_stored[self.mb_buckets_year] = pd.DataFrame(
                    self.mb_buckets_np[:, :-1].copy(),
                    index=self.buckets_grid_point_label,
                    columns=self.buckets[:-1],)

        # store snowline
        if self.store_snowline:
            if self.mb_buckets_year not in self._snowline_year:
                number_buckets = buckets_month + self._snowline_start_month
                # + 1 is for the snow bucket
                not_melted_layers = np.any(
                    self.mb_buckets_np[:, :number_buckets + 1] > 0, axis=1)

                # check if completely snow free
                if not np.any(not_melted_layers):
                    self._snowline.append(np.inf)
                    # store a height value for snow free case
                    if np.inf not in self.snowline_inf_values:
                        self.snowline_inf_values[np.inf] = np.max(heights) + 50
                # check if fully snow covered
                elif np.all(not_melted_layers):
                    self._snowline.append(-np.inf)

                    if -np.inf not in self.snowline_inf_values:
                        # store a height value for fully snow covered case
                        self.snowline_inf_values[-np.inf] = np.min(heights) - 50
                # otherwise get the lowest elevation band with snow cover
                else:
                    self._snowline.append(heights[not_melted_layers][-1])
                self._snowline_year.append(self.mb_buckets_year)

        # nothing to return as every thing is stored is some variables
        return None

    def _bucket_aging(self):
        """In this function the aging of buckets happens.

        This means all buckets are moved one step and an empty fresh snow bucket
        is added at the top. The last bucket is the ice bucket and it is always
        set to zero before calling the aging in
        _apply_climate_step_and_aging_to_buckets (but we also check for this).
        """

        mb_buckets_np = self.mb_buckets_np

        # check that all snow and firn buckets are positive
        if mb_buckets_np[:, :-1].min() < 0:
            raise ValueError("Bucket values must all be positive.")

        # check ice bucket is empty and we do not loose any mass
        if np.any(mb_buckets_np[:, -1]):
            raise ValueError("Ice bucket not empty, need to check!")

        mb_buckets_val_old = mb_buckets_np[:, :-1]
        len_h = len(self.buckets_grid_point_label)
        # add a new fresh snow bucket and move everything to the next older
        # bucket, the last firn bucket is transferred to ice here
        updated_bucket_np = np.concatenate(
            [np.zeros(len_h).reshape(len_h, 1), mb_buckets_val_old], axis=1)

        self.mb_buckets_np = updated_bucket_np

    def _run_until(self, heights, year, mb_resolution):

        # check the current year of the bucket has not exceeded the current year
        if self.mb_buckets_year > year and not self.use_previous_mbs:
            raise InvalidWorkflowError(
                "The current buckets are valid for the float year "
                f"{self.mb_buckets_year}, and you want to have values for "
                f"{year}. If you want to get previously calculated mass "
                "balance values set use_previous_mbs to True. Be aware that "
                "in case that previously calculated mb values are available "
                "your provided heights here will be ignored!")

        # calculate all needed time steps with the same heights
        if self.climate_resolution == 'annual':
            if mb_resolution == 'annual':
                missing_float_years = range(self.mb_buckets_year, int(year) + 1)
            else:
                # mb_resolution can not be shorter than climate_resolution
                raise NotImplementedError(f"mb_resolution: {mb_resolution}")
        elif self.climate_resolution == 'monthly':
            if mb_resolution == 'annual':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=year,
                    include_last_year=True, )
            elif mb_resolution == 'monthly':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=np.ceil(year) + 1)
                # only keep those actually needed
                missing_float_years = [yr for yr in missing_float_years
                                       if yr <= year]
            else:
                # mb_resolution can not be shorter than climate_resolution
                raise NotImplementedError(f"mb_resolution: {mb_resolution}")
        elif self.climate_resolution == 'daily':
            if mb_resolution == 'annual':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=year,
                    include_last_year=True, daily=True)
            elif mb_resolution == 'monthly':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=np.ceil(year) + 1,
                    daily=True
                )
                # only keep those actually needed
                y, m = floatyear_to_date(year)
                y_end = y if m != 12 else y + 1
                m_end = m + 1 if m != 12 else 1
                missing_float_years = [yr for yr in missing_float_years
                                       if yr < date_to_floatyear(y_end, m_end)]
            elif mb_resolution == 'daily':
                missing_float_years = float_years_timeseries(
                    y0=np.floor(self.mb_buckets_year), y1=np.ceil(year) + 1,
                    daily=True
                )
                # only keep those actually needed
                y, m, d = floatyear_to_date(year, return_day=True)
                missing_float_years = [yr for yr in missing_float_years
                                       # date_to_floatyear can handle overflows
                                       # e.g. 32.12.2000 == 01.01.2001
                                       if yr < date_to_floatyear(y, m, d + 1)]
            else:
                raise NotImplementedError(f"mb_resolution: {mb_resolution}")
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # if use_previous_mbs=True it could be there are no missing_float_years
        for yr in missing_float_years:
            if yr < self.mb_buckets_year:
                continue

            self._apply_climate_step_and_aging_to_buckets(heights=heights,
                                                          year=yr)

    def get_annual_mb(self,
                      heights: ArrayLike,
                      year: int or float,
                      add_climate: bool = False,
                      climatic_mb_or_ice_mb: str = 'climatic_mb',
                      **kwargs,
                      ) -> np.float64 or tuple:
        """Get annual climatic mass balance or the ice mass balance.

        The ice mass balance is important for ice dynamics.

        Parameters
        ----------
        heights : ArrayLike
            The heights of each grid point for the provided year. They need to
            have the same length as the number of defined grid points and in
            the same order. The number of grid points at its order were defined
            either through fl or hbins at initialisation.
        year : int or float
            Year in calendar float year or as int
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        climatic_mb_or_ice_mb : str, default 'climatic_mb'
            Defines if you want to retrive the climatic mass balance or the mass
            balance only for ice. The later one is meant to be used together
            with ice dynamics
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Annual climatic mass balance or ice mass balance in metres of ice
            per second. If ``add_climate`` is True, also returns mean
            temperature, the sums of melt temperature, total precipitation, and
            solid precipitation.
        """

        # compute all steps up to the desired target year using constant heights
        self._run_until(heights=heights, year=year,
                        mb_resolution='annual')

        if climatic_mb_or_ice_mb == 'climatic_mb':
            mbs = self._climatic_mb
        elif climatic_mb_or_ice_mb == 'ice_mb':
            mbs = self._ice_mb
        else:
            raise NotImplementedError(
                f"'climatic_mb_or_ice_mb': {climatic_mb_or_ice_mb}")

        # ok now everything should be available, and we can sum up annual values
        # as needed
        if self.climate_resolution == 'annual':
            annual_mb = mbs[self._year_to_index[year]]
        elif self.climate_resolution == 'monthly':
            float_months = float_years_timeseries(y0=year, y1=year+1,
                                                  daily=False)[:-1]
            idx = [self._year_to_index[yr] for yr in float_months]
            annual_mb = np.sum(mbs[idx], axis=0)
        elif self.climate_resolution == 'daily':
            float_days = float_years_timeseries(y0=year, y1=year+1,
                                                daily=True)[:-1]
            idx = [self._year_to_index[yr] for yr in float_days]
            annual_mb = np.sum(mbs[idx], axis=0)
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # convert from kg m-2 to m s-1
        rho = cfg.PARAMS['ice_density']
        annual_mb = ((annual_mb - self.mbmod.bias) / self.sec_in_year(year) /
                     rho)

        if add_climate:
            # because of the use of different climate resolutions we always need
            # to make a distinct call here, even it is potentially already
            # called in _apply_climate_step_and_aging_to_buckets
            t, tmelt, prcp, prcpsol = self.mbmod.get_annual_climate(
                heights=self.mb_heights[year], year=year)
            return annual_mb, t, tmelt, prcp, prcpsol
        else:
            return annual_mb

    def get_monthly_mb(self,
                       heights: ArrayLike,
                       year: float,
                       add_climate: bool = False,
                       climatic_mb_or_ice_mb: str = 'climatic_mb',
                       **kwargs,
                       ) -> np.float64 or tuple:
        """Get monthly climatic mass balance or the ice mass balance.

        The ice mass balance is important for ice dynamics.

        Parameters
        ----------
        heights : ArrayLike
            The heights of each grid point for the provided year. They need to
            have the same length as the number of defined grid points and in
            the same order. The number of grid points at its order were defined
            either through fl or hbins at initialisation.
        year : float
            Year in calendar float year
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        climatic_mb_or_ice_mb : str, default 'climatic_mb'
            Defines if you want to retrive the climatic mass balance or the mass
            balance only for ice. The later one is meant to be used together
            with ice dynamics
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Monthly climatic mass balance or ice mass balance in metres of ice
            per second. If ``add_climate`` is True, also returns mean
            temperature, the sums of melt temperature, total precipitation, and
            solid precipitation.
        """

        # compute all steps up to the desired target year using constant heights
        self._run_until(heights=heights, year=year,
                        mb_resolution='monthly')

        if climatic_mb_or_ice_mb == 'climatic_mb':
            mbs = self._climatic_mb
        elif climatic_mb_or_ice_mb == 'ice_mb':
            mbs = self._ice_mb
        else:
            raise NotImplementedError(
                f"'climatic_mb_or_ice_mb': {climatic_mb_or_ice_mb}")

        # ok now everything should be available, and we can sum up monthly
        # values as needed
        if self.climate_resolution == 'annual':
            raise NotImplementedError('You can not get a monthly mb with an '
                                      'annual climate resolution!')
        elif self.climate_resolution == 'monthly':
            monthly_mb = mbs[self._year_to_index[year]]
        elif self.climate_resolution == 'daily':
            y_start, m_start, d_start = floatyear_to_date(year,
                                                          return_day=True)
            float_days = float_years_timeseries(
                y0=year, y1=year + 1, daily=True)[:-1]
            # only keep days larger equal than the start of the month
            float_days = [day for day in float_days
                          if day >= date_to_floatyear(y_start, m_start, 1)]
            # only keep days smaller the start of the next month
            m_end = m_start + 1 if m_start != 12 else 1
            y_end = y_start if m_start != 12 else y_start + 1
            float_days = [day for day in float_days
                          if day < date_to_floatyear(y_end, m_end, 1)]

            idx = [self._year_to_index[yr] for yr in float_days]
            monthly_mb = np.sum(mbs[idx], axis=0)
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # convert from kg m-2 to m s-1
        rho = cfg.PARAMS['ice_density']
        monthly_mb = ((monthly_mb / self.sec_in_month(year=year) -
                       self.mbmod.bias / self.sec_in_year(year)) /
                      rho)

        if add_climate:
            # because of the use of different climate resolutions we always need
            # to make a distinct call here, even it is potentially already
            # called in _apply_climate_step_and_aging_to_buckets
            t, tmelt, prcp, prcpsol = self.mbmod.get_monthly_climate(
                heights=self.mb_heights[year], year=year)
            return monthly_mb, t, tmelt, prcp, prcpsol
        else:
            return monthly_mb

    def get_daily_mb(self,
                     heights: ArrayLike,
                     year: float,
                     add_climate: bool = False,
                     climatic_mb_or_ice_mb: str = 'climatic_mb',
                     **kwargs,
                     ) -> np.float64 or tuple:
        """Get daily climatic mass balance or the ice mass balance.

        The ice mass balance is important for ice dynamics.

        Parameters
        ----------
        heights : ArrayLike
            The heights of each grid point for the provided year. They need to
            have the same length as the number of defined grid points and in
            the same order. The number of grid points at its order were defined
            either through fl or hbins at initialisation.
        year : float
            Year in calendar float year
        add_climate : bool, default False
            Additionally returns mean temperature and the sums of melt
            temperature, total precipitation, and solid precipitation.
            Avoids recalculating climatology later in some workflows,
            e.g. ``run_with_hydro``.
        climatic_mb_or_ice_mb : str, default 'climatic_mb'
            Defines if you want to retrive the climatic mass balance or the mass
            balance only for ice. The later one is meant to be used together
            with ice dynamics
        **kwargs
            Extra arguments passed to subclasses of this method.

        Returns
        -------
        np.ndarray[np.float64] or tuple[np.ndarray]
            Daily climatic mass balance or ice mass balance in metres of ice
            per second. If ``add_climate`` is True, also returns mean
            temperature, the sums of melt temperature, total precipitation, and
            solid precipitation.
        """

        # compute all steps up to the desired target year using constant heights
        self._run_until(heights=heights, year=year,
                        mb_resolution='daily')

        if climatic_mb_or_ice_mb == 'climatic_mb':
            mbs = self._climatic_mb
        elif climatic_mb_or_ice_mb == 'ice_mb':
            mbs = self._ice_mb
        else:
            raise NotImplementedError(
                f"'climatic_mb_or_ice_mb': {climatic_mb_or_ice_mb}")

        # ok now everything should be available
        if self.climate_resolution in ['annual', 'monthly']:
            raise NotImplementedError(
                'You can not get a daily mb with an annual or monthly climate '
                f'resolution! Your climate resolution: {self.climate_resolution}')
        elif self.climate_resolution == 'daily':
            daily_mb = mbs[self._year_to_index[year]]
        else:
            raise NotImplementedError(
                f"'climate_resolution': {self.climate_resolution}")

        # convert from kg m-2 to m s-1
        rho = cfg.PARAMS['ice_density']
        daily_mb = ((daily_mb / SEC_IN_DAY -
                     self.mbmod.bias / self.sec_in_year(year)) /
                    rho)

        if add_climate:
            # because of the use of different climate resolutions we always need
            # to make a distinct call here, even it is potentially already
            # called in _apply_climate_step_and_aging_to_buckets
            t, tmelt, prcp, prcpsol = self.mbmod.get_daily_climate(
                heights=self.mb_heights[year], year=year)
            return daily_mb, t, tmelt, prcp, prcpsol
        else:
            return daily_mb

    def is_year_valid(self, year):
        return self.mbmod.is_year_valid(year)

    def get_ela(self, year=None, **kwargs):
        # an idea to implement get_ela could be to use the previously computed
        # climatic_mb values together with the used heights and do an
        # interpolation to find the height where climatic_mb = 0, see spline_fit
        # in dynamic spinup
        raise NotImplementedError("Getting the ela for SfcTypeTIModel is"
                                  "currently not supported.")








