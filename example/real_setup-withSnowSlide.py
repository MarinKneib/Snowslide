# # Realistic set-up

# %%
from oggm import cfg
from oggm import tasks, utils, workflow, graphics
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from snowslide import oggm_snowslide_compat

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
rgi_ids = ['RGI60-11.03466']  # This is Talefre
# rgi_ids = ['RGI60-11.03638']  # This is Argentiere
# rgi_ids = ['RGI60-15.03734']  # This is Changri Nup
# rgi_ids = ['RGI60-11.03584'] # Tre les Eaux
# rgi_ids = ['RGI60-15.03507'] # AX010
# rgi_ids = ['RGI60-15.03403'] # Ama Dablan

# Either rerun snowslide or read it from already ran data
run_snowslide = 1

if run_snowslide==0:
    # This is the url with snowslide already run!
    base_url = 'https://cluster.klima.uni-bremen.de/~fmaussion/runs/tests_snowslide/alps_gdirs_whypso/'
    gdirs = workflow.init_glacier_directories(rgi_ids, prepro_base_url=base_url, from_prepro_level=3, prepro_border=80)
else:
    # This is the url with loads of data (dhdt, velocities, etc)
    base_url = 'https://cluster.klima.uni-bremen.de/~oggm/gdirs/oggm_v1.6/L3-L5_files/2023.1/elev_bands/W5E5_w_data/'
    gdirs = workflow.init_glacier_directories(rgi_ids, prepro_base_url=base_url, from_prepro_level=3, prepro_border=80)
    # run snowslide 
    workflow.execute_entity_task(oggm_snowslide_compat.snowslide_to_gdir, gdirs)

gdir = gdirs[0]


# Get the path to the gridded data file & open it
with xr.open_dataset(gdir.get_filepath('gridded_data')) as ds:
    ds = ds.load()

# ds.snowslide_1m.where(ds.glacier_mask).plot()
# plt.show()

# We use OGGM for this. These are "binning" variables to 1D flowlines.
#
# Documentation:
# - https://docs.oggm.org/en/stable/generated/oggm.tasks.elevation_band_flowline.html
# - https://docs.oggm.org/en/stable/generated/oggm.tasks.fixed_dx_elevation_band_flowline.html

tasks.elevation_band_flowline(gdir, bin_variables=['snowslide_1m'])
tasks.fixed_dx_elevation_band_flowline(gdir, bin_variables=['snowslide_1m'], preserve_totals=True)

binned_data_file = gdir.get_filepath('elevation_band_flowline', filesuffix='_fixed_dx')
binned_data = pd.read_csv(binned_data_file, index_col=0)

# binned_data.snowslide_1m.plot();
# plt.show()

from oggm.core import massbalance
from oggm.core.massbalance import mb_calibration_from_scalar_mb, mb_calibration_from_geodetic_mb
from snowslide.oggm_snowslide_compat import MonthlyTIAvalancheModel, mb_calibration_from_geodetic_mb_with_avalanches
from oggm.core.massbalance import MonthlyTIModel

# Compare the two mb models:

# Get model geometry
flowline = gdir.read_pickle('inversion_flowlines')[0]

# We need to recalibrate for this glacier
mb_calibration_from_geodetic_mb_with_avalanches(gdir)

# Create the MB models
# This creates and average of the MB model over a certain period
mb_control = MonthlyTIModel(gdir)
mb_ava = MonthlyTIAvalancheModel(gdir)

# Prepare the data
df_control = pd.DataFrame(index=flowline.dx_meter * np.arange(flowline.nx))
df_ava = pd.DataFrame(index=flowline.dx_meter * np.arange(flowline.nx))
for year in range(2000, 2020):
    df_control[year] = mb_control.get_annual_mb(flowline.surface_h, year=year) * cfg.SEC_IN_YEAR * mb_control.rho
    df_ava[year] = mb_ava.get_annual_mb(flowline.surface_h, year=year) * cfg.SEC_IN_YEAR * mb_control.rho

df_control.mean(axis=1).plot(label='Control')
df_ava.mean(axis=1).plot(label='Avalanches')
plt.legend(fontsize = 14)
plt.title('2000-2020 SMB', fontsize = 20)
plt.xlabel('Distance along flowline [m]', fontsize = 16)
plt.ylabel('Annual SMB [mm w.eq/yr]', fontsize = 16)
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12) 

cfg.PARAMS['store_fl_diagnostics'] = True

tasks.run_random_climate(gdir, nyears=100, y0=2009, halfsize=10, seed=0,
                         mb_model_class=massbalance.MonthlyTIModel,
                         output_filesuffix='_control');

tasks.run_random_climate(gdir, nyears=100, y0=2009, halfsize=10, seed=0,
                         mb_model_class=MonthlyTIAvalancheModel,
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
ds_sel_control = ds_fl_control.isel(time=-1).sel(dis_along_flowline=ds_fl_control.dis_along_flowline < 4500)
ds_sel_ava = ds_fl_ava.isel(time=-1).sel(dis_along_flowline=ds_fl_ava.dis_along_flowline < 4500)
initial_cond = ds_fl_control.isel(time=0).sel(dis_along_flowline=ds_fl_control.dis_along_flowline < 4500)

ds_sel_control.bed_h.plot(color='k');
(ds_sel_control.bed_h + ds_sel_control.thickness_m).plot(label='Control');
(ds_sel_ava.bed_h + ds_sel_ava.thickness_m).plot(label='Avalanches');
plt.legend(fontsize = 14)

# %%
initial_cond.thickness_m.plot(label='Initial thickness', color='black', linestyle='--');
ds_sel_control.thickness_m.plot(label='Control');
ds_sel_ava.thickness_m.plot(label='Avalanches');
plt.legend(fontsize = 14);
plt.title('time = 100.0 yr', fontsize = 20)
plt.xlabel('Distance along flowline [m]', fontsize = 16)
plt.ylabel('Glacier thickness [m]', fontsize = 16)
plt.xticks(fontsize=12)  
plt.yticks(fontsize=12) 

# %%
ds_sel_control.ice_velocity_myr.plot(label='Control');
ds_sel_ava.ice_velocity_myr.plot(label='Avalanches');
plt.legend();

plt.show()

# # %%



