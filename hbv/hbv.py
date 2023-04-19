"""
Minimal spatially distributed HBV model.

This module contains a class for the basic HBV-96 model largely following the
wflow_hbv python implementation.

"""

import datetime

import numpy as np
import pandas as pd
from numba import jit

from . import climate
from . import utils


class BaseModel(object):
    """Minimal spatially distributed HBV model.

    Attributes:
        dt (int): Timestep [s]
        start_date (datetime): Simulation start datetime
        end_date (datetime): Simulation end datetime
        nx (int): Number of grid cells in east-west direction
        ny (int): Number of grid cells in north-south direction
        yll (float): Northing of lower left corner
        xll (float): Easting of lower left corner
        dx (float): Grid cell spacing (same in both directions) [m]
        mask (ndarray): Array marking inside (1) and outside (0) catchment
        elev (ndarray): Cell elevations [m]
        flen (ndarray): Cell distances to (main) outlet [m]
        cell_order (pd.DataFrane): Ordered cell (and downslope cell) indices
        cat_ncells (float): Number of cells in catchment
        sim_dates (list of datetime): Date/time at each timestep of simulation
        date (datetime): Date/time of current timestep

        icf (float): Maximum interception storage [mm]
        lpf (float): Fractional soil moisture limit for aet=pet [-]
        fc (float): Field capacity [mm]
        ttm (float): Melt threshold temperature [K]
        cfmax (float): Snow temperature index melt factor
            [mm K-1 timestep-1]
        cfr (float): Snow refreezing factor [-]
        whc (float): Fractional snow water holding capacity [-]
        beta (float): Soil seepage exponent [-]
        perc (float): Maximum percolation rate through soil [mm timestep-1]
        cflux (float): Maximum capillary flux [mm timestep-1]
        k (float): Coefficient for upper zone outflow [-]
        alpha (float): Exponent for upper zone outflow [-]
        k1 (float): Coefficient for lower zone outflow [-]
        tau (float): Travel speed (number of timesteps to travel 1 m)
            [timestep m-1]
        ssm (float): Minimum slope angle for snow redistribution [degrees]
        ssc (float): Coefficient for exponent in snow holding depth function
        ssa (float): Coefficient for slope in snow holding depth function
        sshdm (float): Minimum holding depth

        lp (float): Soil moisture limit for aet=pet [mm]
        rlag (ndarray): Time lag [timestep(s)]

        nlags (int): Total number of lags (from 0 to maximum lag)
        rlag_3d (ndarray): 3D array for fractional lags at each time lag [timestep(s)]

        out_vars (list of str): Output variable names
        df_cat (pandas.DataFrame): Catchment output time series

        incps (ndarray): Interception storage [mm]
        snws (ndarray): Snowpack solid (ice) water equivalent [mm]
        snwl (ndarray): Snowpack liquid (water) water equivalent [mm]
        sm (ndarray): Soil moisture [mm]
        uz (ndarray): Upper zone storage [mm]
        lz (ndarray): Lower zone storage [mm]
        swe (ndarray): Snowpack total (solid + ice) water equivalent [mm]

        pr (ndarray): Precipitation [mm timestep-1]
        rf (ndarray): Rainfall [mm timestep-1]
        sf (ndarray): Snowfall [mm timestep-1]
        sf_frac (ndarray): Snowfall as fraction of precipitation [-]
        tas (ndarray): Near-surface air temperature [K]
        pet (ndarray): Potential evapotranspiration [mm timestep-1]
        pr_sc (ndarray): Precipitation reaching surface (i.e. after
            interception) [mm timestep-1]
        rf_sc (ndarray): Rainfall reaching surface (i.e. after
            interception) [mm timestep-1]
        sf_sc (ndarray): Snowfall reaching surface (i.e. after
            interception) [mm timestep-1]

        dc_roff (dict): Runoff at catchment outlet by date [mm timestep-1]
        s0 (ndarray): Storage at beginning of timestep
        s1 (ndarray): Storage at end of timestep
        ds (ndarray): Storage change for timestep [mm]

        ss_acc = Ordered cell flow accumulations for avalanche calcs
        ss_yi = Ordered cell y indices for avalanche calcs
        ss_xi = Ordered cell x indices for avalanche calcs
        ss_dsyi = Downstream cell y indices for avalanche calcs
        ss_dsxi = Downstream cell x indices for avalanche calcs
        ss_hd = Ordered cell holding depths for avalanche calcs [mm]

        aet (ndarray): Actual evapotranspiration [mm timestep-1]
        melt (ndarray): Melt [mm timestep-1]
        sm_in (ndarray): Inflow to soil moisture storage [mm timestep-1]
        uz_in (ndarray): Inflow to upper zone storage [mm timestep-1]

        roff_nr (ndarray): Unrouted runoff for timestep [mm timestep-1]
    """

    def __init__(self, **kwargs):
        """Initialise model."""
        self.init_basic(**kwargs)
        self.init_params(**kwargs)
        self.init_outputs(**kwargs)
        self.init_storages()
        self.init_climate_arrays()
        self.init_helper_vars()
        self.init_climate_obj(**kwargs)

    def init_basic(self, **kwargs):
        """Initialise timestep, simulation period, grid and domain definitions.

        Keyword Args:
            dt (int): Timestep [s]
            start_date (datetime): Simulation start datetime
            end_date (datetime): Simulation end datetime
            nx (int): Number of grid cells in east-west direction
            ny (int): Number of grid cells in north-south direction
            yll (float): Northing of lower left corner
            xll (float): Easting of lower left corner
            dx (float): Grid cell spacing (same in both directions) [m]
            mask (ndarray): Array marking inside (1) and outside (0) catchment
            elev (ndarray): Cell elevations [m]
            flen (ndarray): Cell distances to (main) outlet [m]
            cell_order (pd.DataFrame): Ordered cell (and downslope cell) indices
        """
        self.dt = kwargs['dt']
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']
        self.nx = kwargs['nx']
        self.ny = kwargs['ny']
        self.xll = kwargs['xll']
        self.yll = kwargs['yll']
        self.dx = kwargs['dx']
        self.elev = kwargs['elev']
        self.mask = kwargs['mask']
        self.flen = kwargs['flen']
        self.cell_order = kwargs['cell_order']

        self.cat_ncells = np.float32(np.sum(self.mask))

        self.sim_dates = []
        d = self.start_date
        while d <= self.end_date:
            self.sim_dates.append(d)
            d += datetime.timedelta(seconds=self.dt)

        self.date = self.start_date

    def init_params(self, **kwargs):
        """Set parameter values.

        If not updated these parameter values will be fixed throughout the
        simulation. The parameters can be specified as single values or arrays.
        Several derived parameters are calculated from the input (argument)
        parameters (related to routing, as well as soil ET).

        Keyword Args:
            icf (float): Maximum interception storage [mm]
            lpf (float): Fractional soil moisture limit for aet=pet [-]
            fc (float): Field capacity [mm]
            ttm (float): Melt threshold temperature [K]
            cfmax (float): Snow temperature index melt factor
                [mm K-1 timestep-1]
            cfr (float): Snow refreezing factor [-]
            whc (float): Fractional snow water holding capacity [-]
            beta (float): Soil seepage exponent [-]
            perc (float): Maximum percolation rate through soil [mm timestep-1]
            cflux (float): Maximum capillary flux [mm timestep-1]
            k (float): Coefficient for upper zone outflow [-]
            alpha (float): Exponent for upper zone outflow [-]
            k1 (float): Coefficient for lower zone outflow [-]
            tau (float): Travel speed (number of timesteps to travel 1 m)
                [timestep m-1]
            ssm (float): Minimum slope angle for snow redistribution [degrees]
            ssc (float): Coefficient for exponent in snow holding depth function
            ssa (float): Coefficient for slope in snow holding depth function
            sshdm (float): Minimum holding depth

        """
        # Basic parameters
        self.icf = kwargs['icf']
        self.lpf = kwargs['lpf']
        self.fc = kwargs['fc']
        self.ttm = kwargs['ttm']
        self.cfmax = kwargs['cfmax']
        self.cfr = kwargs['cfr']
        self.whc = kwargs['whc']
        self.beta = kwargs['beta']
        self.perc = kwargs['perc']
        self.cflux = kwargs['cflux']
        self.k = kwargs['k']
        self.alpha = kwargs['alpha']
        self.k1 = kwargs['k1']
        self.tau = kwargs['tau']
        self.ssm = kwargs['ssm']
        self.ssc = kwargs['ssc']
        self.ssa = kwargs['ssa']
        self.sshdm = kwargs['sshdm']

        # Derived parameters
        self.lp = self.fc * self.lpf
        self.rlag = (np.max(self.flen) - self.flen) * self.tau
        self.rlag[self.mask == 0] = 0.0
        # ! Modifying (input) cell_order dataframe here currently
        self.cell_order['HD'] = (
                self.ssc * np.exp(self.ssa * self.cell_order['SLOPE'])
        )
        self.cell_order.loc[self.cell_order['HD'] < self.sshdm, 'HD'] = self.sshdm

    def init_outputs(self, **kwargs):
        """Initialise variables controlling and recording simulation outputs.

        Precipitation, ET and runoff will be added to the list if not present.

        Keyword Args:
            out_vars (list of str): Output variable names

        """
        if 'out_vars' in kwargs.keys():
            out_vars = kwargs['out_vars']
            if 'pr' not in out_vars:
                out_vars.append('pr')
            if 'aet' not in out_vars:
                out_vars.append('aet')
            if 'roff' not in out_vars:
                out_vars.append('roff')
        else:
            out_vars = [
                'pr', 'aet', 'melt', 'roff_nr', 'roff', 'swe', 'sm', 'uz', 'lz',
                'incps', 'sca', 'ds', 'mb'
            ]
        self.out_vars = out_vars
        self.df_cat = pd.DataFrame(
            data=0.0,
            index=self.sim_dates,
            columns=self.out_vars
        )

    def init_storages(self):
        """Initialise storage (state variable) arrays.

        SWE is provided for convenience, although it is redundant as both solid
        and liquid components of total snowpack storage are tracked. Initial
        non-zero values for selected storages are assigned here, but these can
        be overriden using self.set_storages().

        """
        self.incps = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.snws = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.snwl = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sm = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.uz = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.lz = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.swe = np.zeros((self.ny, self.nx), dtype=np.float32)

        # Initial storage values where non-zero
        self.sm[:].fill(self.fc)
        self.uz.fill(0.2 * self.fc)
        self.lz.fill(0.33 * self.k1)

    def init_climate_arrays(self):
        """Initialise arrays for climate input fields."""
        self.pr = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.rf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sf_frac = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.tas = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.pet = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.pr_sc = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.rf_sc = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sf_sc = np.zeros((self.ny, self.nx), dtype=np.float32)

    def init_helper_vars(self):
        """Initialise helper variables."""
        # Runoff helper variables
        self.dc_roff = {}  # outflow_date: runoff
        max_lag = int(np.max(np.ceil(self.rlag).astype(np.int)))
        self.nlags = max_lag + 1
        self.rlag_3d = np.zeros((self.nlags, self.ny, self.nx), dtype=np.float32)
        self.rlag_3d -= 999.0
        for lag in range(self.nlags):
            if lag < max_lag:
                self.rlag_3d[lag, :, :] = np.where(
                    (self.rlag >= np.float32(lag)) & (self.rlag < np.float32(lag + 1.0)),
                    1.0 - (self.rlag - np.floor(self.rlag)),
                    self.rlag_3d[lag, :, :]
                )
                self.rlag_3d[lag + 1, :, :] = np.where(
                    (self.rlag >= np.float32(lag)) & (self.rlag < np.float32(lag + 1.0)),
                    self.rlag - np.floor(self.rlag),
                    self.rlag_3d[lag + 1, :, :]
                )
        self.rlag_3d = np.maximum(self.rlag_3d, 0.0)
        self.rlag_3d[:, self.mask == 0] = 0.0

        # Storage and mass balance check helper variables
        self.ds = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.s0 = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.s1 = np.zeros((self.ny, self.nx), dtype=np.float32)

        # For gravitational redistribution, subset on slopes above minimum
        # threshold and convert to numpy arrays (for numba)
        cell_order_sub = self.cell_order.loc[self.cell_order['SLOPE'] > self.ssm]
        self.ss_acc = np.asarray(cell_order_sub['ACC'], dtype=np.int)
        self.ss_yi = np.asarray(cell_order_sub['YI'], dtype=np.int)
        self.ss_xi = np.asarray(cell_order_sub['XI'], dtype=np.int)
        self.ss_dsyi = np.asarray(cell_order_sub['DS_YI'], dtype=np.int)
        self.ss_dsxi = np.asarray(cell_order_sub['DS_XI'], dtype=np.int)
        self.ss_hd = np.asarray(cell_order_sub['HD'], dtype=np.float32)

    def init_climate_obj(self, **kwargs):
        """Initialise user-defined climate input class.

        Purpose is to help pass climate input fields to model at each timestep in
        conjunction with self.get_climate_inputs() method. Both could be overridden.

        """
        self.ci = climate.Climate(
            kwargs['station_details'],
            kwargs['elevation_gradients'],
            kwargs['elev'],
            kwargs['mask'],
            kwargs['ny'],
            kwargs['nx'],
            kwargs['yll'],
            kwargs['xll'],
            kwargs['dx'],
            kwargs['idw_exp']
        )

    def set_storages(self, incps=None, snws=None, snwl=None, sm=None, uz=None,
                     lz=None, swe=None):
        """Specify storage values if needed (e.g. alternative initialisation."""
        if incps is not None:
            self.incps[:] = incps[:]
        if snws is not None:
            self.snws[:] = snws[:]
        if snwl is not None:
            self.snwl[:] = snwl[:]
        if sm is not None:
            self.sm[:] = sm[:]
        if uz is not None:
            self.uz[:] = uz[:]
        if lz is not None:
            self.lz[:] = lz[:]
        if swe is not None:
            self.swe[:] = swe[:]

    def run_model(self):
        """Simulate all timesteps."""
        while self.date <= self.end_date:
            # self.s0 = (
            #         self.incps + self.snws + self.snwl + self.sm + self.uz + self.lz
            # )
            self.simulate_timestep()
            self.date += datetime.timedelta(seconds=self.dt)

    def simulate_timestep(self, pr=None, update_date=False):
        """Simulate one timestep.

        Args:
            pr (ndarray): Precipitation array if passed directly
        """
        self.s0 = (
                self.incps + self.snws + self.snwl + self.sm + self.uz + self.lz
        )

        self.get_climate_inputs(pr)

        # Initialise sub-canopy precipitation (i.e. reaching surface after
        # interception) - will be modified
        self.pr_sc[:] = self.pr[:]
        self.rf_sc[:] = self.rf[:]
        self.sf_sc[:] = self.sf[:]

        self.update_params()

        # ! More frequent avalanches?
        if (self.date.hour == 0) and (self.date.day == 1):
            self.simulate_avalanches()

        self.simulate_interception()
        self.simulate_evapotranspiration()
        self.simulate_snowpack()
        self.simulate_soil_moisture()
        self.simulate_runoff()
        self.simulate_routing()

        self.get_outputs()

        if update_date:
            self.date += datetime.timedelta(seconds=self.dt)

    def get_climate_inputs(self, pr=None):
        """Get climate input arrays.

        Defaults to use Climate object initialised in self.init_climate_obj(),
        which could be overridden alongside this method.
        """
        self.ci.calc_fields(self.date, pr)
        self.pr[:] = self.ci.pr[:]
        self.rf[:] = self.ci.rf[:]
        self.sf[:] = self.ci.sf[:]
        self.tas[:] = self.ci.tas[:]
        self.pet[:] = self.ci.pet[:]

        # Derive snowfall as fraction of precipitation
        self.sf_frac.fill(0.0)
        self.sf_frac[self.pr > 0.0] = self.sf[self.pr > 0.0] / self.pr[self.pr > 0.0]

    def update_params(self):
        """Update parameter values for timestep if needed."""
        pass

    def simulate_interception(self):
        """Simulate canopy interception of precipitation.

        Fill interception storage up to maximum defined by self.icf parameter.
        Use snowfall fraction to partition between snowfall and rainfall.
        """
        if np.max(self.pr) > 0.0:
            incp = np.minimum(self.pr, self.icf - self.incps)
            self.incps += incp
            self.pr_sc -= incp
            self.sf_sc -= (incp * self.sf_frac)
            self.rf_sc = self.pr_sc - self.sf_sc

    def simulate_evapotranspiration(self):
        """Calculate actual evapotranspiration."""
        # Interception evapotranspiration
        incp_et = np.minimum(self.incps, self.pet)
        self.incps -= incp_et

        # Soil evapotranspiration
        soil_pet = (self.pet - incp_et)
        soil_aet = np.where(
            self.sm >= self.lp,
            np.minimum(self.sm, soil_pet),
            np.minimum(soil_pet, self.pet * self.sm / self.lp)  # ! Why not soil_pet in second argument?
        )
        self.sm -= soil_aet

        self.aet = incp_et + soil_aet

    def simulate_snowpack(self):
        """Simulate snowpack accumulation, melt and refreezing.

        Rainfall entering the soil moisture component is added to snow melt
        here too.
        """
        # Potential melt and refreezing (i.e. if unlimited snowpack)
        self.melt = np.where(
            self.tas > self.ttm,
            self.cfmax * (self.tas - self.ttm),
            0.0
        )
        refr = np.where(
            self.tas < self.ttm,
            self.cfmax * self.cfr * (self.ttm - self.tas),
            0.0
        )

        # Limiting to snowpack solid/liquid mass and water-holding capacity
        self.melt = np.minimum(self.melt, self.snws)
        refr = np.minimum(refr, self.snwl)
        self.snws += self.sf_sc + refr - self.melt
        max_snwl = self.snws * self.whc
        self.snwl += self.melt + self.rf_sc - refr

        # Inflow to soil moisture storage
        self.sm_in = np.maximum(self.snwl - max_snwl, 0.0)
        self.snwl -= self.sm_in

        self.swe = self.snws + self.snwl

    def simulate_soil_moisture(self):
        """Simulate soil moisture.

        If field capacity is reached, excess water becomes direct runoff, which
        means that it is passed as an input to the upper zone storage, along
        with seepage through the soil. An adjustment is applied that ensures
        soil moisture is filled to capacity before direct runoff occurs.
        """
        # First estimate of direct runoff
        dir_roff = np.maximum(self.sm + self.sm_in - self.fc, 0.0)
        self.sm += self.sm_in
        self.sm -= dir_roff
        self.sm_in -= dir_roff

        # Seepage to upper zone storage
        seep = np.minimum(self.sm / self.fc, 1.0) ** self.beta * self.sm_in
        self.sm -= seep
        sm_fill = np.minimum(self.fc - self.sm, dir_roff)

        # Update direct runoff and soil moisture balance
        dir_roff -= sm_fill
        self.sm += sm_fill
        self.uz_in = dir_roff + seep

    def simulate_runoff(self):
        """Simulate runoff generation.

        Inflow from soil enters upper zone, from which some percolation to the
        lower zone occurs. Capillary flux returns some water to soil moisture.
        Upper zone outflow calculated using HBV-96 approach (rather than more
        complicated additional option in wflow_hbv).
        """
        # Percolation to lower zone
        self.uz += self.uz_in
        perc = np.minimum(self.perc, self.uz - (self.uz_in / 2.0))  # ! /2.0
        self.uz -= perc

        # Capillary flux
        cap_flux = self.cflux * ((self.fc - self.sm) / self.fc)
        cap_flux = np.minimum(self.uz, cap_flux)
        cap_flux = np.minimum(self.fc - self.sm, cap_flux)
        self.uz -= cap_flux
        self.sm += cap_flux

        # Upper zone outflow
        uz_out = np.minimum(
            np.where(
                perc < self.perc,  # ! no quickflow generated if percolation is below max
                0.0,
                self.k * (self.uz - np.minimum(self.uz_in / 2.0, self.uz))
                ** (1.0 + self.alpha)  # ! /2.0
            ),
            self.uz
        )
        self.uz -= uz_out  # ! Check vs wflow

        # Lower zone inflow/outflow
        self.lz += perc
        lz_out = np.minimum(self.lz, self.k1 * self.lz)
        self.lz -= lz_out

        # Unrouted runoff
        self.roff_nr = uz_out + lz_out

    def simulate_routing(self):
        """Simulate runoff routing.

        Adapted from: https://gmd.copernicus.org/articles/13/6093/2020/ . The
        method is based on calculating a time lag for runoff to reach the
        catchment outlet depending on the distance of a cell from the outlet.
        So for each time lag, add unrouted runoff to the total runoff on the
        relevant outflow date, which is stored in the self.dc_roff dictionary.
        In this implementation, the time lag is split between the lower and
        upper bounding integers according to its fractional component (e.g.
        for a time lag of 1.25 timesteps, 75% of the unrouted runoff is
        assigned to lag=1 and 25% to lag=2.
        """
        roff_r = self.rlag_3d * self.roff_nr
        for lag in range(self.nlags):
            lag = int(lag)
            roff_date = self.date + datetime.timedelta(seconds=(lag * self.dt))
            roff_t = np.sum(roff_r[lag, :, :]) / self.cat_ncells
            if roff_date not in self.dc_roff.keys():
                self.dc_roff[roff_date] = roff_t
            else:
                self.dc_roff[roff_date] += roff_t

    def simulate_avalanches(self):
        """Calculate gravitational snow redistribution based on SnowSlide method.

        Calls a function to permit faster calculation using numba package.
        """
        self.swe, self.snwl, self.snws = grav_redist(
            self.ss_acc, self.ss_yi, self.ss_xi, self.ss_dsyi, self.ss_dsxi,
            self.swe, self.snwl, self.snws, self.ss_hd
        )

    def check_mb(self, pr, aet, roff):
        """Check catchment mass balance.

        Note that catchment runoff needs to be subtracted from self.ds, because
        unrouted runoff is added to storage at end of time step (i.e. self.s1)
        as routing/channel storage effectively, but it is not removed there
        when outflow occurs.
        """
        self.s1 = (
                self.incps + self.snws + self.snwl + self.sm + self.uz + self.lz
                + self.roff_nr
        )
        self.ds = self.s1 - self.s0
        ds_cat = utils.spatial_mean(self.ds, self.mask) - roff
        self.mb = pr - aet - roff - ds_cat
        return(ds_cat)

    def get_outputs(self):
        """Calculate and store catchment outputs in dataframe.

        Mass balance check is called here to avoid duplicate calculations of
        spatial means.
        """
        out_vals = []

        pr = utils.spatial_mean(self.pr, self.mask)
        aet = utils.spatial_mean(self.aet, self.mask)
        roff = self.dc_roff[self.date]

        ds_cat = self.check_mb(pr, aet, roff)

        for var in self.out_vars:
            if var == 'pr':
                out_vals.append(pr)
            elif var == 'aet':
                out_vals.append(aet)
            elif var == 'roff':
                out_vals.append(roff)
            elif var == 'ds':
                out_vals.append(ds_cat)
            elif var == 'sca':
                # ! Hardcoded threshold (assuming swe in mm)
                sca = (
                        np.sum(self.mask[np.logical_and(self.mask == 1, self.swe > 0.1)])
                        / self.cat_ncells
                )
                out_vals.append(sca)
            elif var == 'mb':
                out_vals.append(self.mb)
            else:
                out_vals.append(utils.spatial_mean(getattr(self, var), self.mask))
        idx = self.sim_dates.index(self.date)
        self.df_cat.iloc[idx] = out_vals


@jit(nopython=True)
def grav_redist(ss_acc, ss_yi, ss_xi, ss_dsyi, ss_dsxi, swe, snwl, snws, ss_hd):
    """Calculate gravitational snow redistribution using SnowSlide method."""
    for idx in range(ss_acc.shape[0]):
        yi = ss_yi[idx]
        xi = ss_xi[idx]
        if swe[yi, xi] > 0.0:
            ds_yi = ss_dsyi[idx]
            ds_xi = ss_dsxi[idx]
            liq_frac = snwl[yi, xi] / swe[yi, xi]
            gr = swe[yi, xi] - ss_hd[idx]
            gr_liq = gr * liq_frac
            gr_sol = gr - gr_liq
            if gr > 0.0:
                snwl[yi, xi] -= gr_liq
                snws[yi, xi] -= gr_sol
                snwl[ds_yi, ds_xi] += gr_liq
                snws[ds_yi, ds_xi] += gr_sol
                swe[yi, xi] -= gr
                swe[ds_yi, ds_xi] += gr
    return(swe, snwl, snws)
