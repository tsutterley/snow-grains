#!/usr/bin/env python
u"""
modis_snow_grain_mosaic.py
Written by Tyler Sutterley (10/2020)

Read MODIS Multi-Angle Implementation of Atmospheric Correction (MAIAC)
    Land Surface Bidirectional Reflectance Factor (BRF) grids
    and make daily mosaics of snow grain size

MODIS Aqua/Terra MAIAC BRF Documentation:
    https://lpdaac.usgs.gov/products/mcd19a1v006/
    https://lpdaac.usgs.gov/documents/110/MCD19_User_Guide_V6.pdf

COMMAND LINE OPTIONS:
    --help: list the command line options
    -U X, --user X: username for NASA Earthdata Login
    -N X, --netrc X: path to .netrc file for alternative authentication
    -Y X, --year X: years to run
    -D X, --directory X: working data directory
    -B X, --bounds X: Grid Bounds (xmin,xmax,ymin,ymax)
    -S X, --spacing X: Grid Spacing (dx,dy)
    -P X, --projection X: Grid spatial projection (EPSG code or PROJ4 code)
    -V, --verbose: Verbose output of processing run
    -M X, --mode X: Permission mode of files created

PYTHON DEPENDENCIES:
    numpy: Scientific Computing Tools For Python
        https://numpy.org
        https://numpy.org/doc/stable/user/numpy-for-matlab-users.html
    pyhdf: Python interface for the Hierarchal Data Format 4 (HDF4) library
        http://hdfeos.org/software/pyhdf.php
    netCDF4: Python interface to the netCDF C library
        https://unidata.github.io/netcdf4-python/netCDF4/index.html
    shapely: PostGIS-ish operations outside a database context for Python
        http://toblerity.org/shapely/index.html

UPDATE HISTORY:
    Written 10/2020
"""
from __future__ import print_function

import sys
import os
import re
import netrc
import pyproj
import netCDF4
import pyhdf.SD
import argparse
import datetime
import posixpath
import numpy as np
import osgeo.osr, osgeo.gdal
import shapely.geometry
import snow_grains.utilities

# PURPOSE: create daily mosaics of snow grain size from MODIS imagery
def modis_snow_grain_mosaic(DIRECTORY, YEARS=None, BOUNDS=None, SPACING=None,
    PROJECTION=None, VERBOSE=False, MODE=0o775):
    """
    Create daily mosaics of snow grain size from MODIS imagery
    """
    # server archive and opendap url with MODIS data
    HOST = ['https://ladsweb.modaps.eosdis.nasa.gov','archive','allData','6']
    OPENDAP = ['https://ladsweb.modaps.eosdis.nasa.gov','opendap','allData','6']
    # netCDF4 OpenDAP variables to subset
    vars = ','.join(['Snow_Grain_Size','XDim','YDim','Orbits',
        'grid1km_eos_cf_projection'])
    # find years on host
    pattern = '|'.join(map(str,YEARS)) if YEARS else ''
    remote_years,_ = snow_grains.utilities.lpdaac_list([*HOST,'MCD19A1.json'],
        build=False, pattern=pattern, sort=True)
    # for each year of interest
    for Y in remote_years:
        # find available days of the year
        URL = [*HOST,'MCD19A1','{0}.json'.format(Y)]
        remote_days,_ = snow_grains.utilities.lpdaac_list(URL,
            build=False, pattern='\d+', sort=True)
        # build a mosaic for each day of the year
        for D in remote_days:
            URL = [*HOST,'MCD19A1',Y,'{0}.json'.format(D)]
            files,lastmod = snow_grains.utilities.lpdaac_list(URL, build=False)
            for f in files:
                # download netCDF4 bytes into memory
                fileurl = [*OPENDAP,'MCD19A1',Y,D,'{0}.nc?{1}'.format(f,vars)]
                remote_buffer = snow_grains.utilities.from_lpdaac(fileurl,
                    build=False, verbose=VERBOSE)
                # read MODIS netCDF4 file from OpenDAP
                X,Y,GPS,SGS,crs = read_MODIS_netCDF4(remote_buffer)

# PURPOSE: read MODIS HDF4 file and extract snow grain size
def read_MODIS_HDF4(filename):
    """
    Read MODIS MAIAC HDF4 file and extract snow grain size and coordinates
    """
    # open HDF file for reading
    fileID = pyhdf.SD.SD(filename, pyhdf.SD.SDC.READ)

    # HDF file attributes
    attr = fileID.attributes()
    # read and extract attributes
    StructMetadata = {}
    StructPath = []
    # for each line of structured metadata
    for a in attr['StructMetadata.0'].splitlines():
        if re.search('(?<!END_)GROUP\=(.*?)$',a):
            # find start of group
            GROUP = re.search('(?<!END_)GROUP\=(.*?)$',a).group(1)
            StructPath.append(GROUP)
        elif re.search('END_GROUP\=(.*?)$',a):
            # find end of group
            GROUP = re.search('END_GROUP\=(.*?)$',a).group(1)
            StructPath.remove(GROUP)
        elif re.search('(.*?)\=(.*?)$',a):
            # create structure path within dictionary
            temp = StructMetadata
            for key in StructPath:
                # add key to dictionary
                if key not in temp.keys():
                    temp[key] = {}
                # ascend list
                temp = temp[key]
            # find keyed metadata
            s = re.search('(.*?)\=(.*?)$',a)
            # convert values from string to type if possible
            try:
                temp[s.group(1).strip()] = eval(s.group(2).strip())
            except NameError:
                temp[s.group(1).strip()] = s.group(2).strip()

    # extract projection information
    projection = StructMetadata['GridStructure']['GRID_1']['Projection']
    radius,_,_,_,central_longitude,_,false_easting,false_northing,_,_,_,_,_ = \
        StructMetadata['GridStructure']['GRID_1']['ProjParams']
    # create PROJ4 projection for grid
    kwds = dict(radius=radius, longitude_natural_origin=central_longitude,
        false_easting=false_easting, false_northing=false_northing)
    proj_string = ('+proj=sinu +lon_0={longitude_natural_origin} '
        '+R={radius} +x_0={false_easting} +y_0={false_northing}')
    crs = pyproj.CRS.from_string(proj_string.format(**kwds))

    # extract grid information
    XDim = StructMetadata['GridStructure']['GRID_1']['XDim']
    YDim = StructMetadata['GridStructure']['GRID_1']['YDim']
    UpperLeft = StructMetadata['GridStructure']['GRID_1']['UpperLeftPointMtrs']
    LowerRight = StructMetadata['GridStructure']['GRID_1']['LowerRightMtrs']
    # construct X and Y coordinates
    dX = (LowerRight[0] - UpperLeft[0])/XDim
    dY = (LowerRight[1] - UpperLeft[1])/YDim
    X = UpperLeft[0] + dX*np.arange(XDim)
    Y = UpperLeft[1] + dY*np.arange(YDim)

    # extract time coordinates and convert to GPS time
    orbit_time_stamps = attr['Orbit_time_stamp'].split()
    rx = re.compile('((\d{4})(\d{3})(\d{2})(\d{2}))(T|A)',re.VERBOSE)
    GPS = np.zeros_like(orbit_time_stamps,dtype=np.float)
    for i,t in enumerate(orbit_time_stamps):
        s = rx.match(t).group(1)
        UNIX = snow_grains.utilities.get_unix_time(s,format='%Y%j%H%M')
        GPS[i] = snow_grains.utilities.convert_delta_time(UNIX,
            epoch1=(1970,1,1,0,0,0), epoch2=(1980,1,6,0,0,0))

    # extract snow grain size
    dset = fileID.select('Snow_Grain_Size')
    attrs = dset.attributes()
    SGS = np.ma.array(dset.get(), fill_value=attrs['_FillValue'], dtype='f')
    # set data mask
    SGS.mask = (SGS.data == SGS.fill_value)
    # valid data points
    ii,jj,kk = np.nonzero(~SGS.mask)
    # add offset and multiply by scale factor to valid values
    SGS.data[ii,jj,kk] += attrs['add_offset']
    SGS.data[ii,jj,kk] *= attrs['scale_factor']

    # return the dimensions, snow grain size and coordinates
    return (X,Y,GPS,SGS,crs)

# PURPOSE: read MODIS netCDF file from OpenDAP and extract snow grain size
def read_MODIS_netCDF4(filename):
    """
    Read MODIS MAIAC netCDF4 file and extract snow grain size and coordinates
    """
    # open HDF file for reading
    fileID = netCDF4.Dataset(filename.filename, 'r', memory=filename.read())

    # extract projection information
    grid1km_eos_cf_projection = fileID.variables['grid1km_eos_cf_projection']
    projection = grid1km_eos_cf_projection.grid_mapping_name
    radius = grid1km_eos_cf_projection.earth_radius
    central_longitude = grid1km_eos_cf_projection.longitude_of_central_meridian
    # create PROJ4 projection for grid
    kwds = dict(radius=radius, longitude_natural_origin=central_longitude)
    proj_string = '+proj=sinu +lon_0={longitude_natural_origin} +R={radius}'
    crs = pyproj.CRS.from_string(proj_string.format(**kwds))

    # construct X and Y coordinates
    X = fileID.variables['XDim']
    Y = fileID.variables['YDim']

    # extract time coordinates and convert to GPS time
    orbit_time_stamps = fileID.Orbit_time_stamp.split()
    rx = re.compile('((\d{4})(\d{3})(\d{2})(\d{2}))(T|A)',re.VERBOSE)
    GPS = np.zeros_like(orbit_time_stamps,dtype=np.float)
    for i,t in enumerate(orbit_time_stamps):
        s = rx.match(t).group(1)
        UNIX = snow_grains.utilities.get_unix_time(s,format='%Y%j%H%M')
        GPS[i] = snow_grains.utilities.convert_delta_time(UNIX,
            epoch1=(1970,1,1,0,0,0), epoch2=(1980,1,6,0,0,0))

    # extract snow grain size
    SGS = np.ma.array(fileID.variables['Snow_Grain_Size'][:,:], dtype=np.float,
        fill_value=fileID.variables['Snow_Grain_Size']._FillValue)
    # set data mask
    SGS.mask = (SGS.data == SGS.fill_value)

    # close the netCDF4 file
    fileID.close()
    # return the dimensions, snow grain size and coordinates
    return (X,Y,GPS,SGS,crs)

def main():
    # Read the system arguments listed after the program
    parser = argparse.ArgumentParser(
        description="""Read MODIS Multi-Angle Implementation of Atmospheric
            Correction (MAIAC) Land Surface Bidirectional Reflectance Factor
            (BRF) grids and make daily mosaics of snow grain size
            """
    )
    # command line parameters
    parser.add_argument('--user','-U',
        type=str, default='',
        help='Username for NASA Earthdata Login')
    parser.add_argument('--netrc','-N',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        help='Path to .netrc file for authentication')
    # output working data directory
    parser.add_argument('--directory','-D',
        type=lambda p: os.path.abspath(os.path.expanduser(p)),
        default=os.getcwd(),
        help='Working data directory')
    # years of data to run
    parser.add_argument('--year','-Y',
        type=int, nargs='+',
        help='Years to run')
    # mosaic spatial parameters
    parser.add_argument('--bounds','-B',
        type=float, nargs=4, metavar=('xmin','xmax','ymin','ymax'),
        help='Grid bounds')
    parser.add_argument('--spacing','-S',
        type=float, nargs=2, metavar=('dx','dy'),
        help='Grid spacing')
    # spatial projection (EPSG code or PROJ4 string)
    parser.add_argument('--projection','-P',
        type=str, default='4326',
        help='Spatial projection as EPSG code or PROJ4 string')
    # verbosity settings
    parser.add_argument('--verbose','-V',
        default=False, action='store_true',
        help='Verbose output of run')
    # permissions mode of the local files (number in octal)
    parser.add_argument('--mode','-M',
        type=lambda x: int(x,base=8), default=0o775,
        help='permissions mode of output files')
    args = parser.parse_args()

    # NASA Earthdata hostname
    URS = 'urs.earthdata.nasa.gov'
    # get authentication
    if not args.user and not args.netrc:
        # check that NASA Earthdata credentials were entered
        args.user=builtins.input('Username for {0}: '.format(URS))
        # enter password securely from command-line
        PASSWORD=getpass.getpass('Password for {0}@{1}: '.format(args.user,URS))
    elif args.netrc:
        args.user,LOGIN,PASSWORD=netrc.netrc(args.netrc).authenticators(URS)
    else:
        # enter password securely from command-line
        PASSWORD=getpass.getpass('Password for {0}@{1}: '.format(args.user,URS))
    # build an opener for LP.DAAC
    snow_grains.utilities.build_opener(args.user, PASSWORD)

    # recursively create directory if presently non-existent
    if not os.access(args.directory, os.F_OK):
        os.makedirs(args.directory, args.mode)

    #-- check internet connection before attempting to run program
    if snow_grains.utilities.check_connection('https://lpdaac.usgs.gov'):
        # run program with parameters
        modis_snow_grain_mosaic(args.directory, YEARS=args.year,
            BOUNDS=args.bounds, SPACING=args.spacing,
            PROJECTION=args.projection, VERBOSE=args.verbose,
            MODE=args.mode)

# run main program
if __name__ == '__main__':
    main()
