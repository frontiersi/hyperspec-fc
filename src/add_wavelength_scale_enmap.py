from osgeo import gdal
import glob

gdal.UseExceptions()

def read_enmap_wavelengths(csv_file):
    """
    Reads a CSV file containing wavelengths information for ENMAP data.

    Parameters:
    csv_file (str): The path to the CSV file with three columns: band, wavelength, and fwhm.

    Returns:
    tuple: Three lists containing bands, wavelengths, and fwhms respectively.
    """
    with open(csv_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip().split(',') for line in lines[1:]]
    bands = [int(line[0]) for line in lines]
    wavelengths = [float(line[1]) for line in lines]
    fwhms = [float(line[2]) for line in lines]
    return bands, wavelengths, fwhms

def set_band_descriptions(geotiff_path, band_names):
    """
    Sets the band descriptions in a GeoTIFF file using the provided band names.

    Parameters:
    geotiff_path (str): The path to the GeoTIFF file.
    band_names (list): A list of band names to describe each raster band in the GeoTIFF.

    Raises:
    ValueError: If the file cannot be opened or if the number of band names does not match
                the number of raster bands.
    """
    ds = gdal.Open(geotiff_path, gdal.GA_Update)
    if ds is None:
        raise ValueError(f"Could not open the GeoTIFF file: {geotiff_path}")
    
    num_bands = ds.RasterCount
    if len(band_names) != num_bands:
        raise ValueError("The number of band names must match the number of raster bands.")
    
    for i, band_name in enumerate(band_names, start=1):
        band = ds.GetRasterBand(i)
        band.SetDescription(str(band_name))
        band.SetScale(0.0001)

    del ds  # Close and save changes to the dataset

def main():
    # Read wavelengths from the CSV file
    bands, wavelengths, fwhms = read_enmap_wavelengths('enmap_wavelengths.csv')

    # Find all files matching the pattern ENMAP*.tif
    enmap_files = glob.glob('ENMAP*.tif')
    
    # Set band descriptions for each file
    for enmap_file in enmap_files:
        set_band_descriptions(enmap_file, wavelengths)

if __name__ == '__main__':
    main()
