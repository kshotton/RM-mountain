import numpy as np
import numpy.ma as ma


def read_asc(file_path, data_type=np.float32, return_metadata=True):
    """Read ascii raster and return array (optionally with metadata)."""
    # Headers
    dc = {}
    with open(file_path, 'r') as fh:
        for i in range(6):
            line = fh.readline()
            key, val = line.rstrip().split()
            dc[key] = val
    nx = int(dc['ncols'])
    ny = int(dc['nrows'])
    xll = float(dc['xllcorner'])
    yll = float(dc['yllcorner'])
    dx = float(dc['cellsize'])
    dy = float(dc['cellsize'])
    nodata = float(dc['NODATA_value'])
    
    # Values array
    arr = np.loadtxt(file_path, dtype=data_type, skiprows=6)
    arr = ma.masked_values(arr, nodata)
    
    if return_metadata:
        return(arr, nx, ny, xll, yll, dx, dy)
    else:
        return(arr)


def write_asc(array, file_path, fmt, nx, ny, xll, yll, dx, nodata_value):
    """Write ascii raster to file."""
    headers = (
        'ncols         ' + str(nx) + '\n' +
        'nrows         ' + str(ny) + '\n' +
        'xllcorner     ' + str(xll) + '\n' +
        'yllcorner     ' + str(yll) + '\n' +
        'cellsize      ' + str(dx) + '\n' +
        'NODATA_value  ' + str(nodata_value)
    )
    if ma.isMaskedArray(array):
        output_array = array.filled(nodata_value)
    else:
        output_array = array
    np.savetxt(file_path, output_array, fmt=fmt, header=headers, comments='')


def spatial_mean(val_arr, mask_arr):
    """Calculate spatial mean of values where mask == 1."""
    sm = np.mean(val_arr[mask_arr==1])
    return(sm)


def interpolate(
        station_values, station_details, elevation_gradient, elev, mask, ny, nx, yll, xll, dx, idw_exp,
        simulation_only, adjustment_method, bounds=None
):
    """
    Interpolate a field using inverse distance weighting (optional elevation gradients).

    Elevation gradients are applied using a similar approach to MicroMet: (1) adjust point values
    to a reference elevation, (2) interpolate adjusted values, (3) reintroduce elevation signal
    via elevation gradients.

    Args:
        station_values (numpy.ndarray): Values in same order as stations are listed in station_details
        station_details (pd.DataFrame): Dataframe of station metadata/file paths
        elevation_gradient (float): Elevation gradient per metre increase in elevation
        elev (ndarray): Cell elevations [m]
        mask (ndarray): Array marking inside (1) and outside (0) catchment
        ny (int): Number of grid cells in north-south direction
        nx (int): Number of grid cells in east-west direction
        yll (float): Northing of lower left corner
        xll (float): Easting of lower left corner
        dx (float): Grid cell spacing (same in both directions) [m]
        idw_exp (float): Exponent for IDW weight calculations
        simulation_only (bool): Flag to use only those stations for which Use_In_Simulation field
            of station_details is equal to 1
        adjustment_method (int): Use 1 for elevation adjustment of temperature-like variables and
            2 for precipitation-like variables (see elevation_adjustment() function).
        bounds (tuple): Minimum and maximum permitted values in interpolated array. If None then no
            check is made on plausibility

    """
    # Subset stations just on those to be used in simulation
    if simulation_only:
        station_details = station_details.loc[station_details['Use_In_Simulation'] == 1].copy()
    stations = station_details['Station'].tolist()

    # Initialise (2d) arrays for interpolate variable
    interp_values = np.zeros((ny, nx), dtype=np.float32)

    # Calculate station weights
    station_weights = calc_idw_weights(station_details, interp_values, ny, yll, xll, dx, idw_exp)

    # Reference elevation (taken as catchment mean elevation)
    ref_elev = np.around(np.mean(elev[mask == 1]))

    # Adjust station values to reference elevation
    station_vals_ref = {}
    i = 0
    for index, row in station_details.iterrows():
        station = row['Station']
        station_elev = row['Elevation']
        station_vals_ref[station] = elevation_adjustment(
            station_values[i], elevation_gradient, station_elev, ref_elev, adjustment_method
        )
        i += 1

    # Interpolate adjusted station values
    for station in stations:
        interp_values += (station_weights[station] * station_vals_ref[station])

    # Apply elevation gradients (i.e. adjust from reference elevation to actual (DEM) elevations)
    interp_values = elevation_adjustment(
        interp_values, elevation_gradient, ref_elev, elev, method=adjustment_method
    )

    # Check bounds
    if bounds is not None:
        interp_values[interp_values < bounds[0]] = bounds[0]
        interp_values[interp_values > bounds[1]] = bounds[1]

    return interp_values


def calc_idw_weights(station_details, interp_values, ny, yll, xll, dx, idw_exp):
    # Weights based on distance and a decay function/parameter (i.e. IDW)
    # - based on arrays of distance from each station (cell) to each other cell
    # - one weights array per station
    station_weights = {}
    for index, row in station_details.iterrows():
        station = row['Station']
        # yi = row['YI']
        # xi = row['XI']
        yi = ny - np.ceil((row['Northing'] - yll) / dx)
        xi = np.floor((row['Easting'] - xll) / dx)
        dist = distmat_v2(interp_values, (yi, xi))
        dist[dist == 0.0] = 0.000001  # account for zero distance at station
        station_weights[station] = 1.0 / (dist ** idw_exp)

    # Normalise station weights so sum of weights is one
    # - simplifies IDW calculations
    stations = station_details['Station'].tolist()
    for station in stations:
        if stations.index(station) == 0:
            sum_weights = station_weights[station].copy()
        else:
            sum_weights += station_weights[station]
    for station in stations:
        station_weights[station] /= sum_weights

    return station_weights


def distmat_v2(a, index):
    """Calculate distance of a point from all points in 2d array.

    https://stackoverflow.com/questions/61628380/calculate-distance-from-all-points-in-numpy-array-to-a-single-point-on-the-basis
    """
    i, j = np.indices(a.shape, sparse=True)
    return np.sqrt((i - index[0]) ** 2 + (j - index[1]) ** 2, dtype=np.float32)


def elevation_adjustment(x, gradient, elevation, target_elevation, method):
    """Adjust a value from its elevation to a target elevation.

    Method (1) is for temperature-like variables. Method (2) follows the CRHM
    approach to precipitation adjustment.

    Args:
        x (float or ndarray): Value/array to adjust
        gradient (float): Gradient for adjustment
        elevation (float or ndarray): Elevation associated with value
        target_elevation (float or ndarray): Target elevation to adjust to
        method (int): Flag to indicate form of function to use

    """
    if method == 1:
        x_target = x + gradient * (target_elevation - elevation)
    elif method == 2:
        x_target = x * (1.0 + gradient * (target_elevation - elevation) / 100.0)
        x_target = np.maximum(x_target, 0.0)
    return x_target
