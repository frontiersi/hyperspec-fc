#!/bin/bash

# Check if a file name was provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <input_file.tif>"
  exit 1
fi

# Input TIFF file
INPUT_TIFF="$1"

# Output TIFF file
OUTPUT_TIFF="${INPUT_TIFF%.*}_BSQ.tif"

# Run gdal_translate to compress, tile, and reformat the TIFF
gdal_translate -of GTiff \
  -co COMPRESS=DEFLATE \
  -co ZLEVEL=9 \
  -co TILED=YES \
  -co BLOCKXSIZE=64 \
  -co BLOCKYSIZE=64 \
  -co INTERLEAVE=BAND \
  -co NUM_THREADS=ALL_CPUS \
  "$INPUT_TIFF" "$OUTPUT_TIFF"

# Check that gdal_translate was successful
if [ $? -ne 0 ]; then
  echo "Error: gdal_translate failed."
  exit 1
fi

# Add compressed and tiled overviews to the output TIFF
gdaladdo -r average \
  --config COMPRESS_OVERVIEW DEFLATE \
  --config ZLEVEL_OVERVIEW 9 \
  --config TILED_OVERVIEW YES \
  "$OUTPUT_TIFF" 4 8 16 32 64

# Check that gdaladdo was successful
if [ $? -ne 0 ]; then
  echo "Error: gdaladdo failed."
  exit 1
fi

echo "Processing complete. Output file: $OUTPUT_TIFF"