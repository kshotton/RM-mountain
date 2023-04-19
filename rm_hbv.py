import os
import sys
import datetime
import itertools

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.optimize

from rmwspy import gcopula_sparaest as sparest
from rmwspy import random_mixing_whittaker_shannon as rmws
from hbv import hbv
from hbv import utils

import matplotlib.pylab as plt


def transform_coordinates(p_en, ys, xs, yinc, xinc):
    # Get gauge locations on "standard" (0, 1, ...) grid
    p_xy = np.copy(p_en)
    p_xy[:,0] = (p_xy[:,0] - ys)/yinc
    p_xy[:,1] = (p_xy[:,1] - xs)/xinc
    p_xy = p_xy.astype(int)
    return p_xy


def exponential_model(distance, sill, range_):
    return sill * (np.exp(-distance / range_)) + (1.0 - sill)


def fit_copula_lsq(
        p_xy, prec, outputfile=None, n_subsets=500, random_seed=None, bounds=([0.0, 0.0], [1.0, 100000000.0])
):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Sample from time series
    if n_subsets < prec.shape[1]:
        idx = np.random.choice(prec.shape[1], n_subsets)
        prec_sample = prec[:, idx]
    else:
        prec_sample = prec
    
    # Station separation distances
    separation_distances = []
    for i, j in itertools.combinations(range(p_xy.shape[0]), 2):
        distance = ((p_xy[i][0] - p_xy[j][0]) ** 2 + (p_xy[i][1] - p_xy[j][1]) ** 2) ** 0.5
        separation_distances.append(distance)
    separation_distances = np.array(separation_distances)
    
    # Kendall's tau correlations
    kendalls_tau = []
    for i, j in itertools.combinations(range(p_xy.shape[0]), 2):
        tau, _ = st.kendalltau(prec_sample[i, :], prec_sample[j, :])
        kendalls_tau.append(tau)
    kendalls_tau = np.array(kendalls_tau)
    
    # Conversion of Kendall's tau to Pearson correlations
    correlations = np.sin(kendalls_tau * np.pi / 2.0)
    
    # Fit covariance function via least squares
    parameters, _ = scipy.optimize.curve_fit(exponential_model, separation_distances, correlations, bounds=bounds)
    sill, range_ = parameters
    
    # String representation of cmod for RMWSPy
    cmod = '0.01 Nug(0.0) + %1.12f' % (sill) + ' Exp(%1.12f)' % (range_) # ***** added 0.01 Nug(0.0) term *****
    
    return cmod


def fit_copula_mle(
        p_xy, prec, covmods='Exp', ntries=6, n_in_subset=5, outputfile=None, use_timeseries_fitting=True,
        n_subsets=500, random_seed=None,
):
    if random_seed is not None:
        np.random.seed(random_seed)

    # Jitter to reduce influence of zeros and other ties
    # - sensitivity testing suggested that it is better to generate a random jitter for every observation, (i.e. each
    # timestep and gauge), rather than one set of jitters for each timestep
    prec = prec.copy()
    jitters = np.random.uniform(-0.0001, 0.0001, prec.shape[0] * prec.shape[1])
    jitters = jitters.reshape(prec.shape)
    prec += jitters

    # Transformation to copula (rank) space (separately for each gauge)
    u = (st.rankdata(prec, axis=1) - 0.5) / prec.shape[1]

    # Covariance functions that will be tried in the fitting (omit Matern (at least) with small number of gauges)
    if isinstance(covmods, str):
        covmods = [covmods]

    # New flags for MC are "use_timeseries_fitting" (to use modified fitting approach based on sampling all gauges at
    # n_subsets timesteps) and "n_subsets" (number of subsets (timesteps) to use in fitting)
    cmods = sparest.paraest_multiple_tries(
        np.copy(p_xy),
        u,
        ntries=[ntries, ntries],  # tries with (a) timeseries subsets and (b) initial parameters
        n_in_subset=n_in_subset,  # number of values in subsets (= number of gauges in MC case)
        # neighbourhood='nearest',  # subset search algorithm
        covmods=covmods,  # covariance functions
        outputfile=outputfile,  # store all fitted models in an output file
        use_timeseries_fitting=use_timeseries_fitting,  # flag to use modified fitting approach for MC
        n_subsets=n_subsets,  # number of subsets (timesteps) to use in fitting
    )

    # Take the copula model with the highest likelihood and reconstruct model string from parameter array
    likelihood = -666
    for model in range(len(cmods)):

        for tries in range(ntries):
            if cmods[model][tries][1]*-1. > likelihood:

                likelihood = cmods[model][tries][1] * -1.
                cmod = '0.01 Nug(0.0) + %1.3f %s(%1.3f)' % (
                    cmods[model][tries][0][1], covmods[model],  cmods[model][tries][0][0]
                )
                # print(likelihood)
                print(cmod)

                if covmods[model] == 'Mat':
                    cmod += '^%1.3f' % (cmods[model][tries][0][2])

    return cmod


def generate_field(
        p_xy, prec, p0, ysize, xsize, cmod, gamma_shape, gamma_scale, include_elevation, p0_zadj,
        gamma_scale_zadj, random_seed=None
):
    # Assumes that prec is passed in as a 1D array (one value per gauge for the timestep being simulated)

    if random_seed is not None:
        np.random.seed(random_seed)

    # Convert marginal distribution parameters to arrays (if not already)
    # - creating new variables so as not to mess things up elsewhere (may not be necessary)
    if not isinstance(p0, np.ndarray):
        _p0 = np.repeat(p0, prec.shape[0])
    else:
        _p0 = p0
    if not isinstance(gamma_scale, np.ndarray):
        _gamma_scale = np.repeat(gamma_scale, prec.shape[0])
    else:
        _gamma_scale = gamma_scale
    _gamma_shape = np.repeat(gamma_shape, prec.shape[0])

    # Transform observations to standard normal using the fitted cdf
    # - zero (dry) observations
    mp0 = prec < 0.1  # mp0 = prec == 0.0  # TODO: Keith/David to confirm change
    lecp = p_xy[mp0]  # lecp are the coords of gauges with obs P = 0
    lecv = np.ones(lecp.shape[0]) * st.norm.ppf(_p0[mp0])  # lecv are the less or equal constraint values, i.e. obs that recorded zero P
    # - non-zero (wet) observations
    cp = p_xy[~mp0]  # cp are the equality constraint coords, i.e. gauges with non-zero P
    x = prec[~mp0]  # x = prec[prec >= 0.1]  # apply wet/dry threshold
    cv = st.norm.ppf(
        (1. - _p0[~mp0])
        * st.gamma.cdf(x, _gamma_shape[~mp0], scale=_gamma_scale[~mp0])
        + _p0[~mp0]
    )  # cv are the equality constraint non-zero values

    # Initialize and run Random Mixing Whittaker-Shannon field simulation
    CS = rmws.RMWS(
        domainsize=(ysize, xsize),
        covmod=cmod,
        nFields=1,
        cp=cp,
        cv=cv,
        le_cp=lecp,
        le_cv=lecv,
        optmethod='no_nl_constraints',
        minObj=0.4,  # orig value = 0.4
        maxbadcount=20,  # orig value = 20
        maxiter=100,  # orig value = 300
    )
    CS()

    # Backtransform simulated fields to original data space
    f_prec_field = st.norm.cdf(CS.finalFields)
    f_prec_field = f_prec_field[0, :, :]
    f_prec_field = np.flipud(f_prec_field)
    if include_elevation:
        mp0f = f_prec_field <= p0_zadj
        f_prec_field[mp0f] = 0.0
        f_prec_field[~mp0f] = (f_prec_field[~mp0f] - p0_zadj[~mp0f]) / (1. - p0_zadj[~mp0f])
        f_prec_field[~mp0f] = st.gamma.ppf(f_prec_field[~mp0f], gamma_shape, scale=gamma_scale_zadj[~mp0f])
    else:
        # f_prec_fields = st.norm.cdf(CS.finalFields)
        mp0f = f_prec_field <= p0
        f_prec_field[mp0f] = 0.0
        f_prec_field[~mp0f] = (f_prec_field[~mp0f]-p0)/(1.-p0)
        f_prec_field[~mp0f] = st.gamma.ppf(f_prec_field[~mp0f], gamma_shape, scale=gamma_scale)

        # save simulated precipitation fields
        # np.save('sim_precfields_tstp=%i.npy'%tstep, f_prec_fields)

        # random index for plotting single realization
        # rix = np.random.randint(0, f_prec_field.shape[0], 1)

        # basic plots
        # plot single realization
        # plt.figure()
        # plt.imshow(f_prec_field[rix[0]], origin='lower', interpolation='nearest', cmap='viridis')
        # plt.plot(p_xy[:,1],p_xy[:,0],'x',c='black')
        # plt.xlabel('no. of 50 m grid squares', fontsize=10)
        # plt.ylabel('no. of 50 m grid squares', fontsize=10)
        # plt.title('Spatial precipitation field', fontsize=10)
        # plt.colorbar().set_label(label="precipitation (mm)", size=10)
        # # plt.savefig('C:/ProgramData/Water_Blade_Programs/keith/rm-hbv-master_3_elev_distns/working/annual_ascii_fields/prec_field.png')
        # plt.savefig('C:/Users/b1043453/OneDrive - Newcastle University/OnePlanet PhD/RM_HBV/rm-hbv-master_3_elev_distns_nans/working/example_pr_field_plots/example_pr_field.png')
        # # plt.savefig(r'prec_field_tstp=%i.png'%tstep)
        # plt.clf()
        # plt.close()

    return f_prec_field


def simulate_member(
        start_date, n_timesteps, hbv_setup_dict, p_xy, prec, p0, ysize, xsize, cmod, gamma_shape, gamma_scale,
        output_folder, member_id, obs_flows, time_groups, include_elevation, p0_zadj, gamma_scale_zadj,
        random_seed=None, q=None
):

    if random_seed is not None:
        np.random.seed(random_seed)

    # HBV model for spin-up
    hbv_spinup_model = hbv.BaseModel(**hbv_setup_dict)
    hbv_spinup_model.run_model()

    # HBV model for simulation - apply final states from spin-up model as initial conditions
    hbv_model = hbv.BaseModel(**hbv_setup_dict)
    hbv_model.set_storages(incps=hbv_spinup_model.incps,
                           snws=hbv_spinup_model.snws,
                           snwl=hbv_spinup_model.snwl,
                           sm=hbv_spinup_model.sm,
                           uz=hbv_spinup_model.uz,
                           lz=hbv_spinup_model.lz,
                           swe=hbv_spinup_model.swe)

    # Simulate each timestep
    for idx in range(n_timesteps):
        print(member_id, idx)
        current_date = start_date + datetime.timedelta(days=idx)
        marginal_time_group = time_groups['marginal_time_group'][idx]
        copula_time_group = time_groups['copula_time_group'][idx]
        precip_field = generate_field(
            p_xy, prec[:, idx], p0[marginal_time_group], ysize, xsize, cmod[copula_time_group], 
            gamma_shape[marginal_time_group], gamma_scale[marginal_time_group], include_elevation, 
            p0_zadj[marginal_time_group], gamma_scale_zadj[marginal_time_group],
        )

        # check generated precip_field for NaNs
        # set max number of iterations = 10
        no_of_iters = 10
        for i in range(no_of_iters):
            if np.isnan(precip_field).sum() > 0:
                precip_field = generate_field(
                    p_xy, prec[:, idx], p0[marginal_time_group], ysize, xsize, cmod[copula_time_group], 
                    gamma_shape[marginal_time_group], gamma_scale[marginal_time_group], include_elevation, 
                    p0_zadj[marginal_time_group], gamma_scale_zadj[marginal_time_group],
                )
                if i == no_of_iters - 1:
                    print('field still has NaNs')
            else:
                break

        # save precip_field(s) to ascii file to look at patterns
        # need to read static maps/fixed properties first
        # elev_path = 'C:/ProgramData/Water_Blade_Programs/keith/rm-hbv-master_3_elev_distns/data/mcdem_50m.asc'
        # elev, nx, ny, xll, yll, dx, dy = utils.read_asc(elev_path)
        #
        # utils.write_asc(
        #     precip_field, os.path.join(output_folder, 'precip_field_day_%i_member_%s.asc' %(idx+1, member_id)),
        #     '%.4f', nx, ny, xll, yll, dx, -999.0)

        # precip_field = precip_field[0,:,:]  # moved into generate_field()
        # precip_field = np.flipud(precip_field)  # moved into generate_field()
        hbv_model.simulate_timestep(precip_field, update_date=True)

    # Write HBV output file
    output_path = os.path.join(output_folder, 'hbv_' + str(member_id) + '.csv')
    hbv_model.df_cat.to_csv(output_path, index_label='datetime')

    # Join modelled and observed flows - drop zero flows, which occur everyday from Nov-Apr only
    hbv_model.df_cat.index.name = 'date'
    flows_df = pd.concat([obs_flows.obs_flow_mm, hbv_model.df_cat.roff], axis=1, join="inner")
    flows_df.drop(flows_df[flows_df['obs_flow_mm'] == 0].index, inplace=True)

    # Calculate performance metrics
    nse = calc_nse(flows_df.obs_flow_mm, flows_df.roff)
    rmse = calc_rmse(flows_df.obs_flow_mm, flows_df.roff)
    bias = calc_bias(flows_df.obs_flow_mm, flows_df.roff)
    metrics = (member_id, nse, rmse, bias)

    # Performance metrics are returned if using serial processing but put in a queue for writing if using parallel
    if q is None:
        return metrics
    else:
        q.put(metrics)


def calc_rmse(obs, sim):
    rmse = (np.sum((obs-sim)**2) / len(obs))**0.5
    return rmse


def calc_nse(obs, sim):
    nse = 1.0 - (np.sum((obs - sim)**2))/(np.sum((obs - np.mean(obs))**2))
    return nse


def calc_bias(obs, sim):
    bias = 100.0 * np.sum(sim - obs) / np.sum(obs)
    return bias


def metric_writer(output_path, q=None):
    with open(output_path, 'w') as fh:
        fh.write('Member_ID,NSE,RMSE,BIAS\n')
        while True:
            msg = q.get()
            if msg == 'kill':
                break
            output_line = ','.join(str(item) for item in msg)
            fh.write(output_line + '\n')


# save precip_field(s) to ascii file to look at patterns
# season = 0
# utils.write_asc(
#     precip_field[season], os.path.join(output_folder, 'precip_field.asc'), '%.4f', nx, ny, xll, yll, dx, -999.0
# )
# utils.write_asc(
#     gamma_scale_zadj[season], os.path.join(output_folder, 'gamma_scale_zadj.asc'), '%.4f', nx, ny, xll, yll,
#     dx, -999.0
# )
