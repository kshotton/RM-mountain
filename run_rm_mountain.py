"""
Script for baseline random mixing + HBV simulations.

Some points to note:
    - Marginal distribution(s) is fitted outside of this script currently.
    - Spatial copula may be fitted in this script or it may be specified.
    - Functions are imported from rm_hbv.py to do copula fitting, generate fields and run HBV.

Updates on 14/03/2023:
    - Using "time groups" to generalise e.g. seasons, so now can do annual, seasonal or custom groupings.
    - The time groups can be different for marginal distributions and spatial copulas.
    - Need a new input file to indicate which group each day belongs to (e.g. which season or which intensity band)

"""
import os
import datetime

import numpy as np
import pandas as pd
from sorcery import dict_of

import rm_hbv
from hbv import utils

# =================================================================================================
# USER INPUTS

# -------------------------------------------------------------------------------------------------
# General

wet_threshold = 0.1  # mm (value is dry if < wet_threshold and wet if >= wet_threshold)
station_metadata_path = './data/MCRB_Stations.csv'
input_n_timesteps = 4019  # number of timesteps in climate input files
include_elevation = True  # option to use marginal distributions varying spatially with elevation
marginal_time_grouping = 'annual'  # or 'seasonal'
copula_time_grouping = 'annual' # 'custom'  # use 'custom' for e.g. intensity bands, but can also use 'annual' or 'seasonal'
time_group_path = './data/time_groups.csv'  # needs to match the two arguments above

# -------------------------------------------------------------------------------------------------
# Marginal distribution

# If include_elevation is True we need to put the p0 and gamma_scale into arrays - one
# value for each of the three stations in the same order as they are listed in station_metadata_path
if include_elevation:
    if marginal_time_grouping == 'annual':
        p0 = {0: np.array([0.44, 0.52, 0.61])}
        gamma_shape = {0: 0.628}
        gamma_scale = {0: np.array([8.414, 6.482, 5.993])}
    elif marginal_time_grouping == 'seasonal':
        p0 = {
            0: np.array([0.47, 0.55, 0.60]),
            1: np.array([0.40, 0.51, 0.66]),
            2: np.array([0.37, 0.48, 0.56]),
            3: np.array([0.52, 0.55, 0.63]),
        }
        gamma_shape = {
            0: 0.636,
            1: 0.711,
            2: 0.656,
            3: 0.650,
        }
        gamma_scale = {
            0: np.array([8.184, 6.664, 5.919]),
            1: np.array([4.334, 2.790, 2.791]),
            2: np.array([9.891, 6.525, 5.744]),
            3: np.array([10.376, 8.592, 8.498]),
        }
else:
    if marginal_time_grouping == 'annual':
        p0 = {0: 0.53}
        gamma_shape = {0: 0.628}
        gamma_scale = {0: 6.963}
    elif marginal_time_grouping == 'seasonal':
        p0 = {0: 0.54, 1: 0.52, 2: 0.47, 3: 0.57}
        gamma_shape = {0: 0.636, 1: 0.711, 2: 0.656, 3: 0.650}
        gamma_scale = {0: 6.922, 1: 3.305, 2: 7.387, 3: 9.155}

# If include_elevation is True we also need to specify the elevation gradients for p0 and gamma_scale
# - units should be given per metre increase in elevation
# - this is a negative number for p0 and a positive number for gamma_scale
if include_elevation:
    if marginal_time_grouping == 'annual':
        p0_zgrad = {0: -1.93e-4}
        gamma_scale_zgrad = {0: 2.76e-3}
    elif marginal_time_grouping == 'seasonal':
        p0_zgrad = {0: -1.47e-4, 1: -2.91e-4, 2: -2.14e-4, 3: -1.22e-4}
        gamma_scale_zgrad = {0: 2.57e-3, 1: 1.78e-3, 2: 4.73e-3, 3: 2.16e-3}

# -------------------------------------------------------------------------------------------------
# Copula fitting

fit_copula = True  # False
fitting_seed = 121  # an integer (or numpy.random.SeedSequence().entropy) or None
n_subsets = 1000  # number of timesteps to use
copula_fitting_method = 'lsq'  # or 'mle' - needs to be 'lsq' for e.g. intensity bands (mle = maximum likelihood estimation, lsq = method of least squares)
if not fit_copula:
    if copula_time_grouping == 'annual':
        cmod = {0: '0.01 Nug(0.0) + 0.516 Exp(55.390)'}  # string(s) needs to be provided if not fitting copula
    elif copula_time_grouping == 'seasonal':
        cmod = {
            0: '0.01 Nug(0.0) + 0.499 Exp(54.344)',
            1: '0.01 Nug(0.0) + 0.611 Exp(61.039)',
            2: '0.01 Nug(0.0) + 0.510 Exp(68.349)',
            3: '0.01 Nug(0.0) + 0.443 Exp(46.881)'
        }
    elif copula_time_grouping == 'custom':
        cmod = {
            0: '0.01 Nug(0.0) + 0.787 Exp(61.103)',
            1: '0.01 Nug(0.0) + 0.601 Exp(58.647)',
            2: '0.01 Nug(0.0) + 0.469 Exp(122.294)',
        }

# Grid to cover 5 gauges (FR, UC, HM, K, BV) for copula fitting (eastings and northings)
xs = 623445  # xllcorner
xsize = 400  # ncols
xinc = 50  # cellsize
ys = 5642972  # yllcorner
ysize = 400  # nrows
yinc = 50  # cellsize

# -------------------------------------------------------------------------------------------------
# Simulation (random mixing + HBV)

# Timestep and period
dt = 86400  # time step in seconds (i.e. 86400 = daily)
start_date = datetime.datetime(2005, 10, 1)
end_date = datetime.datetime(2012, 9, 30)
diff = end_date - start_date
simulation_n_timesteps = diff.days + 1  # 365  # or date/time period...

# Grid and static maps (fixed properties)
ysize_sim = 164
xsize_sim = 164
elev_path = './data/mcdem_50m.asc'
mask_path = './data/mcmask.asc'
flen_path = './data/fdis_mc50m.asc'
cell_order_path = './data/MCRB_DS.csv'  # for snow gravitational redistribution

# General
simulation_seed = 121  # an integer (or numpy.random.SeedSequence().entropy) or None
ensemble_size = 200
flow_data_path = './data/env_canada_marmot_creek_flow_data_2005-2012_ymdformat.csv'
use_multiprocessing = True
n_processes = 32
output_folder = './working/annual_200ensemble_lsq_mod2_b16'
metric_output_path = os.path.join(output_folder, 'performance_metrics.csv')

# -----------------------------------------------------------------------------
# HBV-specific climate inputs

# Note that:
# - Elevation gradients for precipitation are only used in spinup currently. In the main simulations the
# precipitation fields are taken directly from the random mixing outputs
# - Station metadata is part of the "general" input section of the script

# Dictionary of elevation gradients by variable and by month
# - pr units are as per CRHM [1/100m]
# - tas units are [K/m]
# - pet units initially assumed to follow pr
# - sign convention:
#       positive = increase with elevation
#       negative = decrease with elevation
elevation_gradients = {
    # adapted from CRHM model code for MCRB (from Github): Julian days converted to Gregorian calendar months
    # Julian day 1 corresponds to 1st January
    'pr': {
        1: 0.0041, 2: 0.0041, 3: 0.0041, 4: 0.01, 5: 0.01, 6: 0.0098,
        7: 0.0098, 8: 0.0098, 9: 0.0059, 10: 0.0059, 11: 0.0041, 12: 0.0041
    },
    # use the monthly mean tas linear regression coefficients from Temp_Regression_monthly_means.ipynb
    'tas': {
       1: -0.00255, 2: -0.00357, 3: -0.00531, 4: -0.00583, 5: -0.00545, 6: -0.00549,
       7: -0.00460, 8: -0.00397, 9: -0.00386, 10: -0.00469, 11: -0.00416, 12: -0.00267
    },
    # there is no positive correlation between PET and elevation so use PET elevation gradient = 0 for each month
    'pet': {
        1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0,
        7: 0.0, 8: 0.0, 9: 0.0, 10: 0.0, 11: 0.0, 12: 0.0
    }
}

# Exponent for inverse-distance weighting interpolation step
idw_exp = 2.0  # also used in elevation-dependent marginal distribution parameters

# -----------------------------------------------------------------------------
# HBV parameters (initial and/or fixed values)

icf = 2.0  # Maximum interception storage [mm]
lpf = 0.9  # Fractional soil moisture limit for aet=pet [-]
fc = 250.0  # Field capacity [mm]
ttm = 273.15  # Melt threshold temperature [K]
cfmax = 3.0  # Snow temperature index melt factor [mm K-1 timestep-1]
cfr = 0.1  # Snow refreezing factor [-]
whc = 0.165224879  # Fractional snow water holding capacity [-]
beta = 2.0  # Soil seepage exponent [-]
perc = 3.0  # Maximum percolation rate through soil [mm timestep-1]
cflux = 2.0  # Maximum capillary flux [mm timestep-1]
k = 0.01  # Coefficient for upper zone outflow [-]
alpha = 0.7  # Exponent for upper zone outflow [-]
k1 = 0.043351686  # Coefficient for lower zone outflow [-]
tau = 1.0 / 86400.0  # Travel speed (number of timesteps to travel 1 m) [timestep m-1]

# For gravitational redistribution of snow
ssm = 25.0  # Minimum slope angle for snow redistribution [degrees]
ssc = 20000.0  # Coefficient for exponent in snow holding depth function
ssa = -0.0546759073  # Coefficient for slope in snow holding depth function
sshdm = 5.0  # Minimum holding depth


# =================================================================================================
# WORKFLOW

# -------------------------------------------------------------------------------------------------
# Read climate station metadata/data

# Read file containing details of stations to use
station_details = pd.read_csv(station_metadata_path)

# Read climate input time series
climate_inputs = {}
for index, row in station_details.iterrows():
    station = row['Station']
    input_path = row['Path']
    climate_inputs[station] = pd.read_csv(
        input_path, index_col='datetime', dtype=np.float32, parse_dates=True, dayfirst=True
    )

# -------------------------------------------------------------------------------------------------
# Put climate data into format for RMWSPY

# Construct array (n_gauges, n_coords=2) of gauge coordinates
station_coords = station_details[['Northing', 'Easting']].values
p_xy = rm_hbv.transform_coordinates(station_coords, ys, xs, yinc, xinc)

# Construct array (n_gauges, n_timesteps) of precipitation observations
n_gauges = station_details['Use_In_Fitting'].sum()
prec = np.zeros((n_gauges, input_n_timesteps))
i = 0
for index, row in station_details.iterrows():
    station = row['Station']
    prec[i,:] = climate_inputs[station]['pr'].values
    i += 1

# Read series of time groups
time_groups = pd.read_csv(
    time_group_path, index_col='datetime', dtype=np.float32, parse_dates=True, dayfirst=True
)

# -------------------------------------------------------------------------------------------------
# Fit spatial copula

# Fit copula
if fit_copula:
    
    # Filter coordinate and precipitation arrays on gauges required in fitting
    p_xy_fitting = p_xy[station_details['Use_In_Fitting'] == 1, :]
    prec_fitting = prec[station_details['Use_In_Fitting'] == 1, :]
    
    # Loop through time groups and fit cmod
    cmod = {}
    for time_group in time_groups['copula_time_group'].unique():
        prec_fitting_sample = prec_fitting[:, time_group == time_groups['copula_time_group']]
        if copula_fitting_method == 'lsq':
            cmod[time_group] = rm_hbv.fit_copula_lsq(
                p_xy_fitting, prec_fitting_sample, n_subsets=n_subsets, random_seed=fitting_seed, 
            )
        elif copula_fitting_method == 'mle':
            cmod[time_group] = rm_hbv.fit_copula_mle(
                p_xy_fitting, prec_fitting_sample, n_subsets=n_subsets, random_seed=fitting_seed,
            )

# -------------------------------------------------------------------------------------------------
# Prepare to initialise HBV

# Read static maps/fixed properties
elev, nx, ny, xll, yll, dx, dy = utils.read_asc(elev_path)
mask = utils.read_asc(mask_path, data_type=int, return_metadata=False)
flen = utils.read_asc(flen_path, return_metadata=False)
cell_order = pd.read_csv(cell_order_path)

# Make HBV setup dictionary (for HBV model initialisation)
# - dict_of creates something like: {'dt': dt, 'start_date': start_date, ...}
setup_dict = dict_of(
    # Timestep, simulation period, grid details, fixed input arrays
    dt, start_date, end_date, nx, ny, xll, yll, dx, mask, elev, flen, cell_order,
    # For climate setup
    station_details, elevation_gradients, idw_exp,
    # Parameters
    icf, lpf, fc, ttm, cfmax, cfr, whc, beta, perc, cflux, k, alpha, k1, tau,
    ssm, ssc, ssa, sshdm,
)

# Also read observed flows and convert from m3/s to mm/day
observed_flows_df = pd.read_csv(flow_data_path, header=0, names=['date', 'flow'])
observed_flows_df['obs_flow_mm'] = observed_flows_df['flow'] * 86400.0 / (9.1 * 1000.0 * 1000.0) * 1000.0
observed_flows_df.set_index('date', inplace=True)
observed_flows_df.index = pd.to_datetime(observed_flows_df.index)  # to match hbv_model.df_cat

# -------------------------------------------------------------------------------------------------
# Interpolate (and account for elevation dependence) in marginal distributions
# - depends on elev array read as part of HBV initialisation

if include_elevation:
    p0_zadj = {}
    gamma_scale_zadj = {}
    for time_group in time_groups['marginal_time_group'].unique():
        p0_zadj[time_group] = utils.interpolate(
            p0[time_group], station_details, p0_zgrad[time_group], elev, mask, ny, nx, yll, xll, dx, idw_exp,
            simulation_only=True, adjustment_method=1, bounds=(0.0, 1.0)
        )
        gamma_scale_zadj[time_group] = utils.interpolate(
            gamma_scale[time_group], station_details, gamma_scale_zgrad[time_group], elev, mask, ny, nx, yll,
            xll, dx, idw_exp, simulation_only=True, adjustment_method=1, bounds=(0.1, 20.0)
        )
else:
    p0_zadj = {time_group: None for time_group in time_groups['marginal_time_group'].unique()}
    gamma_scale_zadj = {time_group: None for time_group in time_groups['marginal_time_group'].unique()}

# -------------------------------------------------------------------------------------------------
# Simulate

# Filter coordinate and precipitation arrays on gauges required in simulation
p_xy_sim = p_xy[station_details['Use_In_Simulation'] == 1, :]
prec_sim = prec[station_details['Use_In_Simulation'] == 1, :]

# Generate random seeds for each ensemble member
if simulation_seed is not None:
    np.random.seed(simulation_seed)
member_seeds = np.random.randint(1000000, 1000000000, ensemble_size)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if __name__ == '__main__':

    if not use_multiprocessing:

        with open(metric_output_path, 'w') as fh:
            fh.write('Member_ID,NSE,RMSE,BIAS\n')

            for member_id in range(1, ensemble_size + 1):
                _, nse, rmse, bias = rm_hbv.simulate_member(
                    start_date, simulation_n_timesteps, setup_dict, p_xy_sim, prec_sim, p0, ysize_sim, xsize_sim, cmod,
                    gamma_shape, gamma_scale, output_folder, member_id, observed_flows_df, time_groups,
                    include_elevation, p0_zadj, gamma_scale_zadj, member_seeds[member_id - 1]
                )
                output_line = ','.join(str(item) for item in [member_id, nse, rmse, bias])
                fh.write(output_line + '\n')

    else:

        import multiprocessing as mp

        manager = mp.Manager()
        q = manager.Queue()
        pool = mp.Pool(n_processes)

        # One process is used to watch for completed simulations so that performance metrics can be written out
        watcher = pool.apply_async(rm_hbv.metric_writer, (metric_output_path, q,))

        # Launch members (up to n_processes members can run at one time)
        jobs = []
        for member_id in range(1, ensemble_size + 1):
            job = pool.apply_async(
                rm_hbv.simulate_member, (
                    start_date, simulation_n_timesteps, setup_dict, p_xy_sim, prec_sim, p0, ysize_sim, xsize_sim, cmod,
                    gamma_shape, gamma_scale, output_folder, member_id, observed_flows_df, time_groups,
                    include_elevation, p0_zadj, gamma_scale_zadj, member_seeds[member_id - 1], q
                )
            )
            jobs.append(job)

        # Tidy up
        for job in jobs:
            job.get()
        q.put('kill')
        pool.close()
        pool.join()

