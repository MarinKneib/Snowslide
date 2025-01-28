"""This module allows the use of snowslide within the OGGM workflow
"""

# Module logger
import logging
log = logging.getLogger(__name__)

# Built ins
import logging
import os

# External libs
import cftime
import numpy as np
import xarray as xr
import pandas as pd

# Locals
import oggm.cfg as cfg
from oggm import utils
from oggm.cfg import SEC_IN_YEAR, SEC_IN_MONTH
from oggm.exceptions import InvalidWorkflowError, InvalidParamsError
from oggm.core.massbalance import MassBalanceModel, mb_calibration_from_scalar_mb, decide_winter_precip_factor, mb_calibration_from_geodetic_mb, MonthlyTIModel
from snowslide.snowslide_main import snowslide_base
from oggm.utils import get_temp_bias_dataframe, clip_scalar
from oggm.core import massbalance

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
def snowslide_to_gdir_meanmonthly_2000_2020(gdir, routing="mfd", Propagation=True, default_grad=-0.0065, t_solid=0, t_liq=2):
    """Add an estimation of avalanches snow redistribution to this glacier directory for the period 2000-2020 using the W5E5 data

    Parameters
    ----------
    gdir : :py:class:`oggm.GlacierDirectory`
        the glacier directory to process
    default_grad: np.float
        temperature gradient as a function of elevation (K/m)
    t_solid: np.float
        temperature below which all the precipitation is solid (°C)
    t_liq: np.float
        temperature above which all the precipitation is liquid (°C)
    """
    # Read W5E5 climate data
    with xr.open_dataset(gdir.get_filepath('climate_historical')) as ds:
        ds_w5e5 = ds.load()

    # monthly temperature and precipitation over the Hugonnet et al. (2021) period (01/2000-01/2020)
    temp = ds_w5e5.temp.sel(time=slice("2000-01", "2020-01")).values
    prcp = ds_w5e5.prcp.sel(time=slice("2000-01", "2020-01")).values

    grad = prcp * 0 + default_grad
    ref_hgt = ds_w5e5.ref_hgt

    # Get minimum and maximum altitude in dem (to compute snow height from precipitation)
    gridded_data_path = gdir.get_filepath("gridded_data")
    with xr.open_dataset(gridded_data_path) as ds:
        ds = ds.load()
    
    # snowslide parameters
    param_routing = {"routing": routing, "preprocessing": True, "compute_edges": True}

    # Get the path of the dem and climate data
    path_to_dem = gdir.get_filepath("dem")

    # define start and end years
    ys = 2000
    ye = 2020
    n_years = ye-ys

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
            rho_freshsnow = 200 # in kg/m3
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

        # delete the variable if it already exists
        if vn in nc.variables:
            nc.remove_variable(vn)

        # create the variable
        v = nc.createVariable(vn, "f4", ("y", "x"), zlib=True)

        # set attributes
        v.units = "m"
        v.long_name = "Snowcover after avalanches"
        # assign data to variable
        v[:] = pfact

def _fallback(gdir):
    """If something wrong happens below"""
    d = dict()
    # Easy stats - this should always be possible
    d["rgi_id"] = gdir.rgi_id
    return d


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
    d["snowslide_deposit_volume_km3"] = np.NaN
    d["melt_f"] = gdir.read_json("mb_calib")['melt_f']
    d["temp_bias"] = gdir.read_json("mb_calib")['temp_bias']
    d["prcp_fac"] = gdir.read_json("mb_calib")['prcp_fac']
    d["melt_f_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['melt_f']
    d["temp_bias_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['temp_bias']
    d["prcp_fac_with_ava"] = gdir.read_json("mb_calib", filesuffix='_with_ava')['prcp_fac']

    try:
        with xr.open_dataset(gdir.get_filepath("gridded_data")) as ds:
            map_result = ds["snowslide_1m"].where(ds["glacier_mask"], np.NaN).load()
            d["snowslide_1m_glacier_average"] = map_result.mean().data
            d["snowslide_deposit_area_km2"] = (
                float(map_result.where(map_result > 1, drop=True).count())
                * resolution**2
                * 1e-6
            )
            d["snowslide_deposit_volume_km3"] = (
                float(map_result.where(map_result > 1, drop=True).sum())
                * resolution**2
                * 1e-9
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
    except:
        return d1, d2

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
    # Get flowline
    flowline = gdir.read_pickle('inversion_flowlines')[0]

    # get mass balance model
    mb_control = MonthlyTIModel(gdir)
    mb_ava = MonthlyTIAvalancheModel(gdir)

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

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)

        mb_annual = np.sum(prcpsol - self.monthly_melt_f * tmelt, axis=1)
        mb_annual = (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
        if add_climate:
            return (
                mb_annual,
                t.mean(axis=1),
                tmelt.sum(axis=1),
                prcp.sum(axis=1),
                prcpsol.sum(axis=1),
            )
        return mb_annual


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

        # these were the 'arbitrary' bounds imposed in Schuster et al., 2023. For our application we allow it to vary more (but keeping the minimum and maximum values)
        #prcp_fac_min = clip_scalar(prcp_fac * 0.8, mi, ma)
        #prcp_fac_max = clip_scalar(prcp_fac * 1.2, mi, ma)
        prcp_fac_min = mi
        prcp_fac_max = ma

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

        prcpsol *= ava_prpc_fac.reshape(prcpsol.shape[0], 1)

        mb_annual = np.sum(prcpsol - self.monthly_melt_f * tmelt, axis=1)
        mb_annual = (mb_annual - self.bias) / SEC_IN_YEAR / self.rho
        if add_climate:
            return (
                mb_annual,
                t.mean(axis=1),
                tmelt.sum(axis=1),
                prcp.sum(axis=1),
                prcpsol.sum(axis=1),
            )
        return mb_annual


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







