# DUSt3R to COLMAP Converter

This tool converts DUSt3R reconstruction outputs into COLMAP format, enabling compatibility with COLMAP-based pipelines and visualization tools. (Like gsplat and InstantSplat)

The utils and scene folder were borrowed from the open source InstantSplat project and the code for dust3r and mast3r was taken from Naver Labs.

## Overview

The conversion process takes DUSt3R's pairwise depth predictions and camera parameters and converts them into COLMAP's sparse reconstruction format. This includes:

1. Camera parameters (intrinsics)
2. Image data and poses (extrinsics)
3. 3D points and their observations
4. Covisibility information

## Requirements

- Python 3.8+
- PyTorch
- NumPy
- OpenCV
- DUSt3R and MAST3R packages

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

```bash
python dust3r2colmap.py \
    --input_dir /path/to/dust3r/output \
    --output_dir /path/to/colmap/output \
    --n_views N \  # Optional: number of views to process
    --depth_threshold 0.1 \  # Optional: threshold for depth consistency
    --max_points 150000  # Optional: maximum number of 3D points
```

## Conversion Process

1. **Loading DUSt3R Output**

   - Reads pairwise depth predictions
   - Loads camera parameters and image information
   - Processes confidence maps

2. **Camera Parameter Conversion**

   - Converts DUSt3R's focal lengths to COLMAP format
   - Processes principal points and image dimensions
   - Creates COLMAP camera models

3. **Point Cloud Generation**

   - Computes covisibility masks between image pairs
   - Filters points based on confidence and depth consistency
   - Generates 3D points with color information

4. **COLMAP Format Writing**
   - Creates COLMAP database structure
   - Writes camera parameters (cameras.txt)
   - Saves image information (images.txt)
   - Stores 3D points (points3D.txt)
   - Generates visualization files (points3D.ply)

## Output Structure

```
output_dir/
├── sparse/
│   ├── cameras.txt    # Camera parameters
│   ├── images.txt     # Image poses and observations
│   ├── points3D.txt   # 3D point coordinates and colors
│   └── points3D.ply   # Visualizable point cloud
└── database.db        # COLMAP database file
```

## Notes

- The conversion preserves DUSt3R's depth and confidence information
- Covisibility masks help ensure consistent reconstruction
- Point cloud density can be controlled via max_points parameter
- Depth threshold affects point cloud quality and density

## Visualization

The converted data can be visualized using COLMAP's GUI:

```bash
colmap gui --database_path /path/to/output/database.db --image_path /path/to/images
```

## Troubleshooting

1. **Memory Issues**

   - Reduce max_points parameter
   - Process fewer views at a time

2. **Quality Issues**

   - Adjust depth_threshold
   - Check input confidence maps
   - Verify camera parameters

3. **Missing Files**
   - Ensure DUSt3R output is complete
   - Check input directory structure
