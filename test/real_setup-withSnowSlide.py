# # Realistic set-up

# %%

from oggm import cfg
from oggm import tasks, utils, workflow, graphics
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# %%
import warnings
# Some annoying warnings sometimes
warnings.filterwarnings(action='ignore', category=UserWarning)

# %% [markdown]
# ## Pick a glacier 

# %%
# Initialize OGGM and set up the default run parameters
cfg.initialize(logging_level='INFO')
dir_path = utils.get_temp_dir('snowslide')
# Local working directory (where OGGM will write its output)
cfg.PATHS['working_dir'] = utils.mkdir(dir_path)

# %%
# rgi_ids = ['RGI60-11.01450']  # This is Aletsch
# rgi_ids = ['RGI60-11.00897']  # This is Hintereisferner
# rgi_ids = ['RGI60-11.03466']  # This is Talefre 
rgi_ids = ['RGI60-11.03638']  # This is Argentiere

# This is the url with snowslide already run!
# base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/runs/tests_snowslide/alps_gdirs_whypso/'

# This is the url with loads of data (dhdt, velocities, etc)
base_url = base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/'

# Can be replaced with 
# https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5 
# for smaller data

# This gets the data for this glacier - download can be a bit long because of all the data
# gdir = workflow.init_glacier_directories(rgi_ids, prepro_base_url=base_url, from_prepro_level=3, prepro_border=80)[0]
gdirs = workflow.init_glacier_directories(rgi_ids, prepro_base_url=base_url, from_prepro_level=3, prepro_border=80)

# %% [markdown]
# Import snowslide as an OGGM task

# %%
from snowslide import oggm_snowslide_compat

# %%
# run the tasks
workflow.execute_entity_task(oggm_snowslide_compat.snowslide_to_gdir, gdirs)

# %%
df = oggm_snowslide_compat.compile_snowslide_statistics(gdirs)

# %%
gdirs

# %% [markdown]
# ## Gridded data

# %%
gdir = gdirs[0]

# %%
# Get the path to the gridded data file & open it
with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
    ds = ds.load()

# %%
ds.snowslide_1m.plot();

# %%
ds.snowslide_1m.where(ds.glacier_mask).plot();

# %%
add = ds.snowslide_1m.where(ds.glacier_mask, drop=True) - 1

# %%
add.plot(cmap='RdBu', vmin=-5, vmax=5);

# %%
ds

# %%
# Map resolution
gdir.grid.dx

# %%
ds.topo.plot();

# %%
ds.glacier_mask.plot();

# %% [markdown]
# Check this tutorial to plot more stuff: https://oggm.org/tutorials/stable/notebooks/10minutes/machine_learning.html

# %% [markdown]
# ### Also works with climate data 

# %%
# Get the path to the climate data file
climate_data_path = gdir.get_filepath('climate_historical')

# Open it
with xr.open_dataset(climate_data_path) as ds_clim:
    ds_clim = ds_clim.load()
    
ds_clim

# %%
ds_clim.prcp.plot();

# %%
ds_clim.temp.plot();

# %%
ds.snowslide_1m

# %% [markdown]
# ## Create an OGGM task to add data to the glacier directory

# %% [markdown]
# See https://github.com/OGGM/oggm/blob/f305390e09a55c1fa204f4d7834c0e377f1c812f/oggm/shop/bedtopo.py#L22 for inspiration
# 
# Here we will just add a dummy mask to the file, to use it later:

# %%
#avalanches = ((ds.topo > 3000) & (ds.topo < 3100) & (ds.glacier_mask == 1)) * 2000
#avalanches.plot();

# %%
# Add it to the dataset and write the file back to disk
#ds['avalanches'] = avalanches
#ds.to_netcdf(gridded_data_path)

# %% [markdown]
# ## Obtain the avalanche data back to the flowlines 

# %% [markdown]
# We use OGGM for this. These are "binning" variables to 1D flowlines. 
# 
# Documentation:
# - https://docs.oggm.org/en/stable/generated/oggm.tasks.elevation_band_flowline.html
# - https://docs.oggm.org/en/stable/generated/oggm.tasks.fixed_dx_elevation_band_flowline.html

# %%
tasks.elevation_band_flowline(gdir, bin_variables=['snowslide_1m', 'millan_v', 'hugonnet_dhdt'])
tasks.fixed_dx_elevation_band_flowline(gdir, bin_variables=['snowslide_1m', 'millan_v', 'hugonnet_dhdt'], preserve_totals=True)

# %% [markdown]
# We just wrote a new file to disk. Let's open it:

# %%
binned_data_file = gdir.get_filepath('elevation_band_flowline', filesuffix='_fixed_dx')
binned_data = pd.read_csv(binned_data_file, index_col=0)
binned_data

# %%
binned_data.millan_v.plot();

# %%
binned_data.hugonnet_dhdt.plot();

# %%
binned_data.snowslide_1m.plot();

# %% [markdown]
# The "dis_along_flowline" variable is not consistent with what OGGM thinks of length (see bug report). This is not a big deal, the length of the data is still correct, so the below works.

# %%
# Note: glacier "length" according to RGI
gdir.read_shapefile('outlines')['Lmax'].iloc[0]

# %% [markdown]
# ## Use this information in the MB model 

# %% [markdown]
# This is a bit far stretched but lets go:

# %%
from oggm.core import massbalance
from oggm.core.massbalance import mb_calibration_from_scalar_mb, mb_calibration_from_geodetic_mb, mb_calibration_from_wgms_mb
from oggm.core.massbalance import MonthlyTIModel_avalanches
from oggm.core.massbalance import MonthlyTIModel

# %%
gdir.get_climate_info()


# %%
gdir.read_json('mb_calib')


# %%
class AvalancheMassBalance(MonthlyTIModel):
    """We Inherit from the standard model, but will add some stuff to it"""

    def __init__(self, *args, **kwargs):
        """ """
        super(AvalancheMassBalance, self).__init__(*args, **kwargs)

    def get_annual_mb(self, heights, year=None, **kwargs):      
        # Here we get the default MB
        smb = super(AvalancheMassBalance, self).get_annual_mb(heights, year=year, **kwargs) 
        
        # Add avalanches
        where_is_avalanche = np.nonzero(binned_data.snowslide_1m.values > 0)
        
        smb[where_is_avalanche] += binned_data.snowslide_1m.values[where_is_avalanche] / cfg.SEC_IN_YEAR / self.rho
        
        # Return
        return smb 

# %% [markdown]
# Compare the two mb models:

# %%
# Get model geometry
flowline = gdir.read_pickle('inversion_flowlines')[0]

# Create the MB models 
# This creates and average of the MB model over a certain period
mb_control = massbalance.MonthlyTIModel_avalanches(gdir)
mb_ava = massbalance.MonthlyTIModel(gdir)

# %%
# Prepare the data
df_control = pd.DataFrame(index=flowline.dx_meter * np.arange(flowline.nx))
df_ava = pd.DataFrame(index=flowline.dx_meter * np.arange(flowline.nx))
for year in range(2000, 2020):
    df_control[year] = mb_control.get_annual_mb(flowline.surface_h, year=year) * cfg.SEC_IN_YEAR * mb_control.rho
    df_ava[year] = mb_ava.get_annual_mb(flowline.surface_h, year=year) * cfg.SEC_IN_YEAR * mb_control.rho

# %%
df_control.plot(legend=False);

# %%
df_ava.plot(legend=False);

# %%
df_control.mean(axis=1).plot(label='Control');
df_ava.mean(axis=1).plot(label='Avalanches');
plt.legend(); plt.title('2000-2020 SMB'); plt.xlabel('Dis along flowline'); plt.ylabel('Annual SMB');

# %% [markdown]
# ## Feed the models to the simulations 

# %%
cfg.PARAMS['store_fl_diagnostics'] = True

# %%
tasks.run_random_climate(gdir, nyears=100, y0=2009, halfsize=10, seed=0,
                         mb_model_class=massbalance.MonthlyTIModel, 
                         output_filesuffix='_control');

# %%
tasks.run_random_climate(gdir, nyears=100, y0=2009, halfsize=10, seed=0,
                         mb_model_class=AvalancheMassBalance, 
                         output_filesuffix='_ava');

# %%
with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix='_control')) as ds:
    ds_avg_control = ds.load()
with xr.open_dataset(gdir.get_filepath('model_diagnostics', filesuffix='_ava')) as ds:
    ds_avg_ava = ds.load()

# %%
ds_avg_control.volume_m3.plot(label='Control');
ds_avg_ava.volume_m3.plot(label='Avalanches');
plt.legend();

# %%
with xr.open_dataset(gdir.get_filepath('fl_diagnostics', filesuffix='_control'), group='fl_0') as ds:
    ds_fl_control = ds.load()
with xr.open_dataset(gdir.get_filepath('fl_diagnostics', filesuffix='_ava'), group='fl_0') as ds:
    ds_fl_ava = ds.load()

# %%
ds_sel_control = ds_fl_control.isel(time=-1).sel(dis_along_flowline=ds_fl_control.dis_along_flowline < 5000) 
ds_sel_ava = ds_fl_ava.isel(time=-1).sel(dis_along_flowline=ds_fl_ava.dis_along_flowline < 5000) 

ds_sel_control.bed_h.plot(color='k');
(ds_sel_control.bed_h + ds_sel_control.thickness_m).plot(label='Control');
(ds_sel_ava.bed_h + ds_sel_ava.thickness_m).plot(label='Avalanches');
plt.legend();

# %%
ds_sel_control.thickness_m.plot(label='Control');
ds_sel_ava.thickness_m.plot(label='Avalanches');
plt.legend();

# %%
ds_sel_control.ice_velocity_myr.plot(label='Control');
ds_sel_ava.ice_velocity_myr.plot(label='Avalanches');
plt.legend();

# %% [markdown]
# ## Things to think about 

# %% [markdown]
# - here we apply avalanching as a constant positive MB - in the future, will the avalanche amounts change?
# - what about the time dependency?
# - importantly, we apply the avalanches without recalibrating the MB. The purpose will be to actually recalibrate the MB with the new information
# - on a glacier per glacier basis we will likely find that influence of avalanches will be small. But at the regional scale, in some regions in the himalayas, I think we can make a difference.
# - lots to think about!

# %%



