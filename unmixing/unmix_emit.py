import numpy as np
from rios import applier, fileinfo, cuiprogress
from scipy.optimize import nnls
import joblib
import os
from osgeo import gdal
import glob

def nnls_unmix(refDataPath, unmixDataPath, 
               pca_file='pca.joblib', 
               vertices_file='optimized_vertices6.npy',
               valid_bands_file='valid_bands.npy',
               sum_to_one=1,
               bare_green_dry=[[1,4], [3,6], [2,5]], # Bare, green, dry endmember mapping
               window_size=512):
    """
    Perform non-negative least squares unmixing on hyperspectral data.
    
    Parameters:
    -----------
    refDataPath : str
        Path to the reflectance data (input)
    unmixDataPath : str
        Path to save the unmixed fractions (output)
    pca_file : str
        Path to the PCA model file
    vertices_file : str
        Path to the optimized vertices file
    valid_bands_file : str
        Path to the valid bands mask file
    sum_to_one : float
        Sum-to-one constraint value
    bare_green_dry : list of lists
        Mapping of endmembers to bare, green, dry categories (0-based indices)
    window_size : int
        Size of processing windows
        
    Returns:
    --------
    dict
        Dictionary containing the output file paths
    """
    gdal.UseExceptions()
    
    # Load required files
    pca = joblib.load(pca_file)
    optimized_vertices = np.load(vertices_file)
    valid_bands = np.load(valid_bands_file)
    
    # Setup weighted endmembers with sum-to-one constraint
    weighted_endmembers = np.vstack((
        optimized_vertices.T, 
        np.ones(optimized_vertices.shape[0]) * sum_to_one
    ))
    
    # Convert to 0-based index
    bare_green_dry = [[int(band)-1 for band in fraction] for fraction in bare_green_dry]
    
    # Setup input/output file associations
    infiles = applier.FilenameAssociations()
    outfiles = applier.FilenameAssociations()
    infiles.nbar = refDataPath
    outfiles.fractions = unmixDataPath
    outfiles.fractions_bgd = unmixDataPath.replace('.tif', '_bgd.tif')
    
    # Setup other arguments
    otherargs = applier.OtherInputs()
    otherargs.noData = fileinfo.ImageInfo(infiles.nbar).nodataval[0]
    otherargs.pca = pca
    otherargs.weighted_endmembers = weighted_endmembers
    otherargs.valid_bands = valid_bands
    otherargs.sum_to_one = sum_to_one
    otherargs.bare_green_dry = bare_green_dry
    
    # Setup processing controls
    controls = applier.ApplierControls()
    controls.windowxsize = controls.windowysize = window_size
    controls.progress = cuiprogress.CUIProgressBar()
    controls.setStatsIgnore(255)
    controls.setOutputDriverName("GTIFF")
    controls.setCreationOptions([
        "COMPRESS=DEFLATE",
        "ZLEVEL=1",
        "BIGTIFF=YES",
        "TILED=YES",
        "INTERLEAVE=BAND",
        "NUM_THREADS=ALL_CPUS",
        f"BLOCKXSIZE={window_size}",
        f"BLOCKYSIZE={window_size}"
    ])
    # controls.setConcurrencyStyle(applier.ConcurrencyStyle(
    #                             numComputeWorkers=4,
    #                             computeWorkerKind=applier.CW_THREADS,
    #                             numReadWorkers=4,
    #                             readBufferInsertTimeout=300, 
    #                             readBufferPopTimeout=300,
    #                             computeBufferPopTimeout=300))
    
    # Apply the unmixing function
    rtn = applier.apply(_nnls_unmix, infiles, outfiles, otherargs, controls=controls)
    
    return rtn

def _nnls_unmix(info, inputs, outputs, otherargs):
    """
    Helper function for NNLS unmixing using RIOS.
    """
    nbar = inputs.nbar[otherargs.valid_bands]
    mask = np.all(nbar == otherargs.noData, axis=0)

    # Flatten reflectance and perform PCA in one step to save memory
    original_shape = nbar.shape
    nbar_transformed = otherargs.pca.transform(nbar.reshape(original_shape[0], -1).T).T

    # Precompute ones array outside the loop
    ones_array = np.full((1, nbar_transformed.shape[1]), otherargs.sum_to_one)

    # Combine weighted reflectance with ones_array using np.vstack
    weightedReflectance = np.vstack((nbar_transformed, ones_array))

    # Initialize fractions array with zeros
    fractions = np.zeros((otherargs.weighted_endmembers.shape[1], nbar_transformed.shape[1]), dtype=np.float32)

    # Use vectorization where possible
    for i in range(weightedReflectance.shape[1]):
        fractions[:, i], _ = nnls(otherargs.weighted_endmembers, weightedReflectance[:, i])

    # Reshape, clip, and convert to 8-bit data
    fractions = np.clip(np.rint(100.0 * fractions.reshape((-1,original_shape[1], original_shape[2]))), 0, 200).astype('uint8')

    # Compute fractions for bare, green, dry vegetation
    fractions_bgd = np.zeros((3, original_shape[1], original_shape[2]), dtype=np.uint8)
    fractions_bgd[0] = fractions[otherargs.bare_green_dry[0]].sum(axis=0)
    fractions_bgd[1] = fractions[otherargs.bare_green_dry[1]].sum(axis=0)
    fractions_bgd[2] = fractions[otherargs.bare_green_dry[2]].sum(axis=0)

    # Apply mask
    fractions_bgd[:, mask] = 255
    fractions[:, mask] = 255

    # Write the output
    outputs.fractions = fractions
    outputs.fractions_bgd = fractions_bgd


# Main workflow
emit_files = glob.glob('/mnt/e/OneDrive - Cibo Labs Pty Ltd/smartsatQuality/**/EMIT_L2A_RFL_*_reflectance.tif', recursive=True)
# Rempve any file with NZ_ in the name
emit_files = [f for f in emit_files if 'NZ_' not in f]
print(f'Found {len(emit_files)} EMIT files.')

for refDataPath in emit_files:
    unmixDataPath = refDataPath.replace('reflectance', 'unmixed')
    print(f'Processing {refDataPath}')

    result = nnls_unmix(
        refDataPath=refDataPath,
        unmixDataPath=unmixDataPath,
        pca_file='../pca.joblib',
        vertices_file='../optimized_vertices6.npy',
        valid_bands_file='../valid_bands.npy',
        sum_to_one=1,
        bare_green_dry=[[1,4], [6], [2,3,5]],  # Bare, green, dry endmember mapping
    )

    print(result.timings.formatReport())