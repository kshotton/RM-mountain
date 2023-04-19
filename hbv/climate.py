# No accounting for missing data currently

import numpy as np
import pandas as pd

from . import utils


class Climate(object):
    """Read and calculate climate inputs.
    
    Approach follows MicroMet: (1) adjust gauge observations to a reference
    elevation using elevation gradients, (2) interpolate gauge observations on
    reference elevation, (3) (re)introduce elevation signal using elevation
    gradients. Main differences are (a) using IDW interpolation rather than
    objective analysis and (b) using CRHM equation for adjusting precipitation
    for elevation. 
    
    Attributes:
        station_details (pd.DataFrame): Dataframe of station metadata/file paths
        elevation_gradients (dict of dicts): Elevation gradients by variable and month
        elev (ndarray): Cell elevations [m]
        mask (ndarray): Array marking inside (1) and outside (0) catchment
        ny (int): Number of grid cells in north-south direction
        nx (int): Number of grid cells in east-west direction
        yll (float): Northing of lower left corner
        xll (float): Easting of lower left corner
        dx (float): Grid cell spacing (same in both directions) [m]
        idw_exp (float): Exponent for IDW weight calculations
        
        pr (ndarray): Precipitation [mm timestep-1]
        rf (ndarray): Rainfall [mm timestep-1]
        sf (ndarray): Snowfall [mm timestep-1]
        tas (ndarray): Near-surface air temperature [K]
        pet (ndarray): Potential evapotranspiration [mm timestep-1]
        
        tm (float): Melting/freezing temperature [K]
        
        stations (list of str): List of station names/IDs
        station_series (dict of pd.DataFrame): Station climate time series
        station_variables (dict of lists): List of variables available at each station
        station_weights (dict of ndarray): Weights arrays for IDW calculations
        ref_elev (float): Reference elevation for interpolation
    """
    
    def __init__(self, station_details, elevation_gradients, elev, mask, ny, nx, yll, xll, dx, idw_exp):
        """
        Args:
            station_details (pd.DataFrame) Dataframe of station metadata/file paths
            elevation_gradients (dict): Dictionary (of dictionaries) containing elevation gradients
            elev (ndarray): Cell elevations [m]
            mask (ndarray): Array marking inside (1) and outside (0) catchment
            ny (int): Number of grid cells in north-south direction
            nx (int): Number of grid cells in east-west direction
            yll (float): Northing of lower left corner
            xll (float): Easting of lower left corner
            dx (float): Grid cell spacing (same in both directions) [m]
            idw_exp (float): Exponent for IDW weight calculations
        """
        # Assign arguments to attributes
        self.station_details = station_details
        self.elevation_gradients = elevation_gradients
        self.elev = elev
        self.ny = ny
        self.nx = nx
        self.yll = yll
        self.xll = xll
        self.dx = dx
        self.idw_exp = idw_exp

        # Subset stations just on those to be used in simulation
        self.station_details = self.station_details.loc[self.station_details['Use_In_Simulation'] == 1]
        
        # Initialise (2d) arrays for climate variables (to be filled each timestep)
        self.pr = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.rf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.sf = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.tas = np.zeros((self.ny, self.nx), dtype=np.float32)
        self.pet = np.zeros((self.ny, self.nx), dtype=np.float32)
        
        # Other attributes
        self.tm = 273.15
        
        # Derived attributes
        self.stations = self.station_details['Station'].tolist()
        
        # Read station time series into a dictionary indexed by station name/ID
        # - also identify variables available for each station
        self.station_series = {}
        self.station_variables = {}
        for index, row in self.station_details.iterrows():
            station = row['Station']
            input_path = row['Path']
            self.station_series[station] = pd.read_csv(
                input_path, index_col='datetime', dtype=np.float32, 
                parse_dates=True, dayfirst=True
            )
            self.station_variables[station] = []
            for variable in ['pr', 'tas', 'pet']:
                if variable in self.station_series[station].columns:
                    self.station_variables[station].append(variable)
        
        # Weights based on distance and a decay function/parameter (i.e. IDW)
        # - based on arrays of distance from each station (cell) to each other cell
        # - one weights array per station
        self.station_weights = utils.calc_idw_weights(
            self.station_details, self.pr, self.ny, self.yll, self.xll, self.dx, self.idw_exp
        )
        
        # Reference elevation (taken as catchment mean elevation)
        self.ref_elev = np.around(np.mean(elev[mask == 1]))

    def calc_fields(self, date, pr=None):
        """Calculate spatial fields of climate inputs for timestep.
        
        Args:
            date (datetime): Date/time of required climate fields
            pr (ndarray): Precipitation array if passed directly
        """
        # Fill climate arrays with zeros
        self.pr.fill(0.0)
        self.rf.fill(0.0)
        self.sf.fill(0.0)
        self.tas.fill(0.0)
        self.pet.fill(0.0)
        
        # Get station values for timestep
        station_vals = {
            'pr': {}, 'tas': {}, 'pet': {}
        }
        for station in self.stations:
            df = self.station_series[station]
            for variable in ['pr', 'tas', 'pet']:
                if variable in self.station_variables[station]:
                    station_vals[variable][station] = (
                        np.float32(df.loc[df.index == date, variable].values[0])
                    )
        
        # Adjust station values to reference elevation
        station_vals_ref = {
            'pr': {}, 'tas': {}, 'pet': {}
        }
        for index, row in self.station_details.iterrows():
            station = row['Station']
            station_elev = row['Elevation']
            for variable in ['pr', 'tas', 'pet']:
                if variable in self.station_variables[station]:
                    if variable == 'tas':
                        method = 1
                    else:
                        method = 2
                    station_vals_ref[variable][station] = utils.elevation_adjustment(
                        station_vals[variable][station],
                        self.elevation_gradients[variable][date.month], 
                        station_elev, self.ref_elev, method
                    )

        # Interpolate adjusted station values
        for station in self.stations:
            if pr is None:
                if 'pr' in self.station_variables[station]:
                    self.pr += (self.station_weights[station] * station_vals_ref['pr'][station])
            if 'tas' in self.station_variables[station]:
                self.tas += (self.station_weights[station] * station_vals_ref['tas'][station])
            if 'pet' in self.station_variables[station]:
                self.pet += (self.station_weights[station] * station_vals_ref['pet'][station])
        
        # Apply elevation gradients (i.e. adjust from reference elevation to
        # actual (DEM) elevations)
        if pr is None:
            self.pr = utils.elevation_adjustment(
                self.pr, self.elevation_gradients['pr'][date.month], self.ref_elev,
                self.elev, method=2
            )
        self.tas = utils.elevation_adjustment(
            self.tas, self.elevation_gradients['tas'][date.month], self.ref_elev,
            self.elev, method=1
        )
        self.pet = utils.elevation_adjustment(
            self.pet, self.elevation_gradients['pet'][date.month], self.ref_elev, 
            self.elev, method=2
        )

        # If precipitation has been passed directly then set it
        if pr is not None:
            self.pr[:] = pr[:]
        
        # Set precipitation below a (low) threshold to zero
        # - could be made a function of timestep
        self.pr[self.pr < 0.01] = 0.0
        
        # Rainfall and snowfall partitioning
        self.rf[:] = np.where(self.tas > self.tm, self.pr, 0.0)
        self.sf[:] = self.pr - self.rf






