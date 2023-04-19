"""
Find the downstream cell for each cell and order by flow accumulation as a
pre-processing step for gravitational redistribution calcs. Save in an output
table, alongside cell slope.
"""

import os
import numpy as np

import utils

# -----------------------------------------------------------------------------
# Paths to input ascii rasters and the output csv file to be created

fdir_path = './data/fdir_mc50m.asc'  # flow direction
facc_path = './data/facc_mc50m.asc'  # flow accumulation
mask_path = './data/mcmask.asc'  # mask
slope_path = './data/Slope_MC50m_QGIS.asc'  # slope [degrees]

output_path = './data/MCRB_DS.csv'

# -----------------------------------------------------------------------------

# Read rasters
fdir, nx, ny, xll, yll, dx, dy = utils.read_asc(fdir_path, data_type=np.int)
facc = utils.read_asc(facc_path, data_type=np.int, return_metadata=False)
mask = utils.read_asc(mask_path, data_type=np.int, return_metadata=False)
slope = utils.read_asc(slope_path, return_metadata=False)

# Populate a dictionary that has flow accumulations for keys and lists for
# values. The lists contain the cell array indices, the downstream cell indices
# and the slope of the cell. Each cell has its own list, which is situated 
# inside a list that contains all cells of a given flow accumulation (i.e.
# as a list of lists)
dc = {}
for yi in range(ny):
    for xi in range(nx):
        if mask[yi,xi] == 1:
            
            # - north
            if fdir[yi,xi] == 64:
                ds_yi = yi - 1
                ds_xi = xi
            # -- north-east
            elif fdir[yi,xi] == 128:
                ds_yi = yi - 1
                ds_xi = xi + 1
            # - east
            elif fdir[yi,xi] == 1:
                ds_yi = yi
                ds_xi = xi + 1
            # -- south-east
            elif fdir[yi,xi] == 2:
                ds_yi = yi + 1
                ds_xi = xi + 1
            # - south
            elif fdir[yi,xi] == 4:
                ds_yi = yi + 1
                ds_xi = xi
            # -- south-west
            elif fdir[yi,xi] == 8:
                ds_yi = yi + 1
                ds_xi = xi - 1
            # - west
            elif fdir[yi,xi] == 16:
                ds_yi = yi
                ds_xi = xi - 1
            # -- north-west
            elif fdir[yi,xi] == 32:
                ds_yi = yi - 1
                ds_xi = xi - 1
            
            # List for cell contains flow accumulation, cell array indices, 
            # downstream cell array indices and cell slope
            inds = [facc[yi,xi], yi, xi, ds_yi, ds_xi, '{0:.2f}'.format(slope[yi,xi])]
            if facc[yi,xi] in dc.keys():
                dc[facc[yi,xi]].append(inds)
            else:
                dc[facc[yi,xi]] = [inds]

# Ensure list of unique flow accumulations does not contain any None entries, 
# which occur outside of the catchment
facc_unq = np.unique(facc)
facc_unq = facc_unq.tolist()
facc_unq = [ind for ind in facc_unq if ind is not None]

# Output file contains one row per cell within the catchment. Columns are flow 
# accumulation, cell array indices, downstream cell array indices and cell slope
with open(output_path, 'w') as fh:
    fh.write('ACC,YI,XI,DS_YI,DS_XI,SLOPE\n')
    for fa in facc_unq:
        for inds in dc[fa]:
            out = ','.join(str(item) for item in inds)
            fh.write(out + '\n')
    
    




