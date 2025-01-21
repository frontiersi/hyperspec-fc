#!/usr/bin/env python

"""
This script provides functions to process NetCDF files and convert them to GeoTIFF format with orthorectification using GDAL. It includes the following functions:
1. extract_variables_and_metadata_to_json(netcdf_file, print_metadata=False):
2. single_image_ortho(img_dat, glt, glt_nodata_value=0):
    Orthorectifies a single image using a geolocation table (GLT).
3. set_band_descriptions(geotiff_path, band_names):
    Sets descriptions for each band in a GeoTIFF file.
4. create_geotiff(output_name, data, transform, projection, nodata_value=None):
    Creates a GeoTIFF file using GDAL with optional NoData value.
5. main(rawargs=None):
    Main function to convert a NetCDF file to GeoTIFF with orthorectification using GDAL.
        input_netcdf (str): Path to the input NetCDF file to convert.
"""

import os
import json
import argparse
import numpy as np
from netCDF4 import Dataset
from osgeo import gdal, osr

# Enable GDAL exceptions for easier error handling
gdal.UseExceptions()


def extract_variables_and_metadata_to_json(netcdf_file, print_metadata=False):
    """
    Extracts specific variables and metadata from a NetCDF file and returns them as a JSON string.
    
    Parameters:
        netcdf_file (str): Path to the NetCDF file.
        print_metadata (bool): If True, prints all available metadata.
        
    Returns:
        str: JSON string containing the extracted data and metadata.
    """
    group_name = 'sensor_band_parameters'
    variables_of_interest = ['wavelengths', 'fwhm', 'good_wavelengths']
    metadata_attributes = ['time_coverage_start', 'time_coverage_end']
    data = {'variables': {}, 'metadata': {}}

    try:
        with Dataset(netcdf_file, 'r') as nc:
            # Extracting variables from the specified group
            sensor_band_params = nc.groups.get(group_name)
            if sensor_band_params:
                for var_name in variables_of_interest:
                    if var_name in sensor_band_params.variables:
                        variable_data = sensor_band_params.variables[var_name][:]
                        data['variables'][var_name] = variable_data.tolist()
                    else:
                        print(f"- Variable '{var_name}' not found in the group.")
            else:
                print(f"The group '{group_name}' does not exist in the file.")

            # Extracting metadata attributes
            for attr_name in metadata_attributes:
                if attr_name in nc.ncattrs():
                    data['metadata'][attr_name] = getattr(nc, attr_name)
                else:
                    print(f"- Metadata attribute '{attr_name}' not found in the file.")
            
            # Print all metadata if requested
            if print_metadata:
                print(f"\n{os.path.basename(netcdf_file)} netadata:")
                for attr_name in nc.ncattrs():
                    print(f"{attr_name}: {getattr(nc, attr_name)}")
                
        # Convert the extracted data and metadata to a JSON string
        return json.dumps(data, indent=2)

    except IOError as e:
        print(f"Failed to open the NetCDF file: {e}")
        return None

def single_image_ortho(img_dat, glt, glt_nodata_value=0):
    """Orthorectify a single image.

    Args:
        img_dat: The image data to be orthorectified.
        glt: Geolocation table (GLT) array.
        glt_nodata_value: Value representing no-data in the GLT.

    Returns:
        Orthorectified image data.
    """
    # Initialize output array with zeros, matching the GLT's spatial dimensions and the image's channel depth
    outdat = np.zeros((glt.shape[0], glt.shape[1], img_dat.shape[-1]), dtype=img_dat.dtype)
    
    # Identify valid GLT entries that are not equal to the no-data value
    valid_glt = np.all(glt != glt_nodata_value, axis=-1)
    
    # Adjust valid GLT indices for zero-based indexing
    glt_valid = glt[valid_glt] - 1
    
    # Map input image data to output based on valid GLT coordinates
    outdat[valid_glt, :] = img_dat[glt_valid[:, 1], glt_valid[:, 0], :]
    
    return outdat

def set_band_descriptions(geotiff_path, band_names):
    """
    Set descriptions for each band in a GeoTIFF file.

    Args:
        geotiff_path: Path to the GeoTIFF file.
        band_names: List of band names to add (as descriptions).
    """
    ds = gdal.Open(geotiff_path, gdal.GA_Update)
    if ds is None:
        raise ValueError(f"Could not open the GeoTIFF file: {geotiff_path}")
    
    num_bands = ds.RasterCount
    if len(band_names) != num_bands:
        raise ValueError("The number of band names must match the number of raster bands.")
    
    for i, band_name in enumerate(band_names, start=1):
        band = ds.GetRasterBand(i)
        band.SetDescription(str(band_name))  # Use the band index (starting at 1)

    del ds  # Close and save changes to the dataset


def create_geotiff(output_name, data, transform, projection, nodata_value=None):
    """Create a GeoTIFF file using GDAL with optional NoData value.

    Args:
        output_name: Path for the output GeoTIFF file.
        data: Image data to be saved.
        transform: GeoTransform to apply.
        projection: Projection information in WKT format.
        nodata_value: Optional NoData value for output bands.
    """
    driver = gdal.GetDriverByName('GTiff')
    bands = data.shape[-1]
    
    # Determine appropriate GDAL data type from NumPy dtype
    if data.dtype == np.uint8:
        dtype = gdal.GDT_Byte
    elif data.dtype == np.int16:
        dtype = gdal.GDT_Int16
    elif data.dtype == np.int32:
        dtype = gdal.GDT_Int32
    elif data.dtype == np.uint16:
        dtype = gdal.GDT_UInt16
    elif data.dtype == np.uint32:
        dtype = gdal.GDT_UInt32
    elif data.dtype == np.float32:
        dtype = gdal.GDT_Float32
    else:
        raise ValueError(f"Unhandled data type: {data.dtype}")

    # Create GeoTIFF dataset with specified options
    dataset = driver.Create(output_name, data.shape[1], data.shape[0], bands, dtype,
                            options=['COMPRESS=DEFLATE',
                                     'TILED=YES',
                                     'BLOCKXSIZE=64',
                                     'BLOCKYSIZE=64',
                                     'ZLEVEL=9',
                                     'NUM_THREADS=ALL_CPUS',
                                     'BIGTIFF=YES',
                                     'INTERLEAVE=BAND'])
    
    # Set the GeoTransform and projection of the dataset
    dataset.SetGeoTransform(transform)
    dataset.SetProjection(projection)
    
    # Write each band of data to the dataset
    for i in range(bands):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(data[:, :, i])
        
        # Assign NoData value if provided
        if nodata_value is not None:
            band.SetNoDataValue(nodata_value)
        
        band.FlushCache()
    
    # Build overviews (pyramids) for the dataset
    gdal.SetConfigOption("COMPRESS_OVERVIEW", "DEFLATE")
    gdal.SetConfigOption("TILED_OVERVIEW", "YES")
    dataset.BuildOverviews('average', [4, 8, 16, 32, 64])
    dataset.FlushCache()

def main(rawargs=None):
    """
    Main function to convert a NETCDF file to GeoTIFF with orthorectification using GDAL.
    Args:
        rawargs (list, optional): List of command-line arguments. If None, arguments will be parsed from sys.argv.
    Raises:
        FileNotFoundError: If the specified output directory does not exist.
        AttributeError: If the output file already exists and overwrite is not allowed.
    Command-line Arguments:
        input_netcdf (str): Path to the input NETCDF file to convert.
        --output_dir (str, optional): Base directory for output files. Defaults to the directory of the input file or the current working directory.
        --overwrite (bool, optional): Flag to overwrite existing files. Defaults to False.
        --nodata (float, optional): NoData value for the output GeoTIFFs. Defaults to None.
        --verbose (bool, optional): Flag to print additional information. Defaults to False.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Convert NETCDF to GeoTIFF with orthorectification using GDAL.")
    parser.add_argument('input_netcdf', type=str, help='File to convert.')
    parser.add_argument('--output_dir', type=str, help='Base directory for output files', default=None)
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing file')
    parser.add_argument('--nodata', type=float, help='NoData value for output GeoTIFFs', default=None)
    parser.add_argument('--verbose', action='store_true', help='Print additional information')
    args = parser.parse_args(rawargs)

    # Set default output directory if not provided
    args.output_dir = args.output_dir or os.path.dirname(args.input_netcdf) or os.getcwd()
    
    # Validate the output directory
    if not os.path.isdir(args.output_dir):
        raise FileNotFoundError(f"Output directory {args.output_dir} does not exist - please create it or try again")

    # Open the input NetCDF file
    nc_ds = Dataset(args.input_netcdf, 'r', format='NETCDF4')

    # Get the metadata and variables of interest from the NetCDF file
    json_data = extract_variables_and_metadata_to_json(args.input_netcdf, bool(args.verbose))
    if json_data:
        wavelengths = json.loads(json_data)['variables'].get('wavelengths', [])
        fwhm = json.loads(json_data)['variables'].get('fwhm', [])
        good_wavelengths = json.loads(json_data)['variables'].get('good_wavelengths', [])    

    # Initialize GLT array with zeros and populate with data from NetCDF
    glt = np.zeros(list(nc_ds.groups['location']['glt_x'].shape) + [2], dtype=np.int32)
    glt[...,0] = np.array(nc_ds.groups['location']['glt_x'])
    glt[...,1] = np.array(nc_ds.groups['location']['glt_y'])

    # Process each variable/dataset in the NetCDF file
    dataset_names = list(nc_ds.variables.keys())
    for ds in dataset_names:
        # Construct output filename
        output_name = os.path.join(args.output_dir, os.path.splitext(os.path.basename(args.input_netcdf))[0] + '_' + ds + '.tif')
        
        # If output file exists and overwrite is not allowed, raise an error
        if os.path.isfile(output_name) and not args.overwrite:
            raise AttributeError(f'File {output_name} already exists. Please use --overwrite to replace')

        # Load data from the current NetCDF variable
        dat = np.array(nc_ds[ds])
        
        # Reshape 2D data to have a single band dimension if necessary
        if len(dat.shape) == 2:
            dat = dat.reshape((dat.shape[0], dat.shape[1], 1))

        # Perform orthorectification on the data
        dat = single_image_ortho(dat, glt)

        # Retrieve GeoTransform and projection information
        geotransform = nc_ds.__dict__["geotransform"]
        spatial_ref = nc_ds.__dict__.get("spatial_ref", None)
        
        # Convert spatial reference to projection string if available
        if spatial_ref:
            srs = osr.SpatialReference()
            srs.ImportFromWkt(spatial_ref)
            projection = srs.ExportToWkt()
        else:
            projection = None
        
        # Create the output GeoTIFF file
        create_geotiff(output_name, dat, geotransform, projection, args.nodata)
        # Add band names to the GeoTIFF file as long as wavelengths are available
        if wavelengths:
            print(f"Adding band names to {output_name}")
            set_band_descriptions(output_name, wavelengths)

# Entry point of the script
if __name__ == "__main__":
    main()
