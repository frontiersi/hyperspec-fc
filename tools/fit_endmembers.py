## Imports

import os
import glob
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
from rios import applier, fileinfo
from netCDF4 import Dataset
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


gdal.UseExceptions()

# Set the location of the proj library
os.environ['PROJ_LIB'] = '/home/pete/miniforge3/envs/kea/share/proj'


"""
Minimal Volume Convex Hull Optimization in 3D

This script finds a minimal-volume convex hull with N vertices that encloses
a given set of 3D points using global optimization (Differential Evolution).

Author: Peter Scarth
Date: 2025-02-18
"""

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Necessary for 3D plotting

def compute_hull_volume(points):
    """
    Compute the volume of the convex hull of given points.

    Parameters:
    - points: ndarray of shape (M, 3).

    Returns:
    - volume: float, volume of the convex hull. Returns np.inf if hull computation fails.
    """
    try:
        hull = ConvexHull(points)
        return hull.volume
    except:
        # If points are coplanar or otherwise degenerate
        return np.inf

def points_inside_hull(original_points, hull_vertices):
    """
    Check which original points are inside the convex hull defined by hull_vertices.

    Parameters:
    - original_points: ndarray of shape (M, 3), points to check.
    - hull_vertices: ndarray of shape (N, 3), vertices of the convex hull.

    Returns:
    - inside: ndarray of shape (M,), boolean array indicating whether each point is inside the hull.
    """
    try:
        delaunay = Delaunay(hull_vertices)
        return delaunay.find_simplex(original_points) >= 0
    except:
        # If the Delaunay triangulation fails (e.g., degenerate hull)
        return np.zeros(len(original_points), dtype=bool)

def define_bounds(original_points, N, padding=1.0):
    """
    Define the bounds for each coordinate of the N new vertices based on the original data.

    Parameters:
    - original_points: ndarray of shape (M, 3), original data points.
    - N: int, number of vertices for the new convex hull.
    - padding: float, extra space added to the bounds.

    Returns:
    - bounds: list of tuples, bounds for each variable in the optimizer.
    """
    mins = original_points.min(axis=0) - padding
    maxs = original_points.max(axis=0) + padding
    bounds = []
    for _ in range(N):
        for dim in range(3):
            bounds.append((mins[dim], maxs[dim]))
    return bounds



def objective(global_variables, original_points, N, penalty_weight=10, pca=None, bound_penalty_weight=1, max_reflectance=1):
    """
    Objective function to minimize: volume of the convex hull + penalties for points outside +
    penalties for inverse-transformed endmembers outside [0, 1].

    Parameters:
    - global_variables: ndarray of shape (3N,), flattened coordinates of N vertices.
    - original_points: ndarray of shape (M, 3), points to be enclosed.
    - N: int, number of vertices for the new convex hull.
    - penalty_weight: float, weight for the penalty term for points outside.
    - pca: PCA object, used for inverse transformation.
    - bound_penalty_weight: float, weight for the penalty term for bound violations.
    - max_reflectance: float, maximum reflectance value for bound violations.

    Returns:
    - total_cost: float, sum of hull volume and penalties.
    """
    # Reshape the variable vector into N x 3 points
    Q = global_variables.reshape((N, 3))
    
    try:
        hull = ConvexHull(Q)
        volume = hull.volume
    except:
        # Assign a large volume if hull computation fails
        return 1e12
    
    # Retrieve the hull vertices for Delaunay triangulation
    hull_vertices = Q[hull.vertices]
    
    # Check which original points are inside the hull
    inside = points_inside_hull(original_points, hull_vertices)
    num_outside = np.size(inside) - np.sum(inside)
    
    # Calculate penalty for points outside
    penalty = penalty_weight * num_outside
    
    # Additional penalty for inverse-transformed endmembers outside [0, Max Reflectance]
    if pca is not None:
        try:
            # Inverse transform to original space
            endmembers = pca.inverse_transform(Q)
            
            # Compute violations: any value <0 or >0.5
            violations = np.sum((endmembers < 0) | (endmembers > max_reflectance))
            
            # Apply penalty for each violation
            penalty += bound_penalty_weight * violations
        except Exception as e:
            # If inverse_transform fails, apply a large penalty
            penalty += 1e6
    
    # Total cost is volume plus penalties
    total_cost = volume + penalty
    
    return total_cost

def find_minimal_convex_hull_global(original_points, N, pca, penalty_weight=10, bound_penalty_weight=1, maxiter=1000, popsize=15, seed=None, max_reflectance=1):
    """
    Find the minimal-volume convex hull enclosing the original points using Differential Evolution.

    Parameters:
    - original_points: ndarray of shape (M, 3), points to enclose.
    - N: int, number of vertices for the new convex hull (must be >=4 for 3D).
    - pca: PCA object, used for inverse transformation.
    - penalty_weight: float, weight for the penalty term for points outside.
    - bound_penalty_weight: float, weight for the penalty term for bound violations.
    - maxiter: int, maximum number of iterations for the optimizer.
    - popsize: int, population size multiplier (default 15).
    - seed: int or None, random seed for reproducibility.

    Returns:
    - optimized_vertices: ndarray of shape (N, 3), optimized positions of hull vertices.
    - result: OptimizeResult object, contains information about the optimization.
    """
    if N < 4:
        raise ValueError("N must be at least 4 to form a 3D convex hull.")
    
    # Define bounds for the optimizer
    bounds = define_bounds(original_points, N, padding=1.0)
    
    # Perform Differential Evolution
    result = differential_evolution(
        func=objective,
        bounds=bounds,
        args=(original_points, N, penalty_weight, pca, bound_penalty_weight, max_reflectance),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=1e-6,
        mutation=(0.5, 1),
        recombination=0.7,
        seed=seed,
        disp=True,
        polish=True,
        init='latinhypercube',
        updating='deferred',  # Improves performance by deferring updates
        workers=1  # Use all available CPU cores
    )
    
    # Extract optimized vertices
    optimized_vertices = result.x.reshape((N, 3))
    return optimized_vertices, result

def plot_convex_hull(original_points, hull_vertices, title='Minimal Volume Convex Hull'):
    """
    Plot the original points and the optimized convex hull in 3D.

    Parameters:
    - original_points: ndarray of shape (M, 3), original data points.
    - hull_vertices: ndarray of shape (N, 3), vertices of the optimized convex hull.
    - title: str, title of the plot.
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot original points
    ax.scatter(original_points[:,0], original_points[:,1], original_points[:,2],
               color='blue', alpha=0.3, label='Original Points')
    
    # Compute convex hull of optimized vertices
    try:
        hull = ConvexHull(hull_vertices)
        for simplex in hull.simplices:
            simplex = np.append(simplex, simplex[0])  # To create a closed loop
            ax.plot(hull_vertices[simplex, 0], hull_vertices[simplex, 1], hull_vertices[simplex, 2],
                    'r-')
        
        # Optionally, plot the vertices
        ax.scatter(hull_vertices[:,0], hull_vertices[:,1], hull_vertices[:,2],
                   color='red', label='Optimized Hull Vertices')
    except Exception as e:
        print("Failed to compute convex hull for visualization:", e)
    
    ax.set_title(title)
    ax.legend()
    
    plt.show()

def plot_endmembers(wavelengths, endmembers, title = ''):
    """
    Plot endmember spectra with wavelengths.
    
    Parameters:
    - wavelengths: ndarray of shape (n_bands,), wavelength values
    - endmembers: ndarray of shape (n_endmembers, n_bands), endmember spectra
    """
    with plt.style.context('dark_background'):
        fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
        
        # Vectorized plotting using numpy operations
        wavelengths_matrix = np.tile(wavelengths, (endmembers.shape[0], 1)).T
        
        # Plot all lines at once with optimized line properties
        lines = ax.plot(wavelengths_matrix, endmembers.T, 
                       linewidth=1.5,  # Optimal line width for performance
                       solid_capstyle='round',
                       antialiased=True)
        
        ax.set_prop_cycle('color', plt.cm.tab10(np.linspace(0, 1, endmembers.shape[0])))
        
        ax.set(xlabel='Wavelength',
               ylabel='Reflectance',
               title=title)
        
        for text in (*ax.get_xticklabels(), *ax.get_yticklabels(),
                    ax.xaxis.label, ax.yaxis.label, ax.title):
            text.set_color('white')
        
        ax.legend([f'Endmember {i+1}' for i in range(endmembers.shape[0])],
                  framealpha=0.8,
                  loc='best',
                  ncol=1)

        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
        fig.tight_layout()


# Load the data
emit_sample = np.load('emit_data.npy')
wavelengths = np.load('wavelengths.npy')
valid_bands = np.load('valid_bands.npy')
sampled_data = np.full((emit_sample.shape[0], len(wavelengths)), -1.0, dtype=np.float32)
sampled_data[:, valid_bands] = emit_sample

# Find the edgemembers
valid_bands = ~np.all(sampled_data < 0, axis=0)
pca = PCA(n_components=3)
pc = pca.fit_transform(sampled_data[:, valid_bands])

# Get at least start_vertices edge members
start_vertices = 30
w_val=0.5
num_vertices = 4
while num_vertices < start_vertices:
    hull = ConvexHull(pc, qhull_options=f'Fx W{w_val} QbB')
    num_vertices = len(hull.vertices)
    w_val -= 0.01

print(f"Found {num_vertices} edge members with w={w_val:.2f}")
edgemembers = pc[hull.vertices]
print(f"Starting with {len(edgemembers)} edge members.")

# Parameters
N = 6 # Number of vertices for the new convex hull (>=4)
penalty_weight = 5.0 # Penalty for points outside the hull
bound_penalty_weight = 0.1 # Penalty for inverse-transformed endmembers outside [0, 1]      
maxiter = 10000000 # Maximum number of iterations for the optimizer
popsize = 15 # Population size multiplier
seed = None # Random seed for reproducibility

# Find minimal convex hull using global optimizer
print(f"Starting optimization to find a convex hull with {N} vertices...")
optimized_vertices, optimization_result = find_minimal_convex_hull_global(
    edgemembers,
    N=N,
    pca=pca,
    penalty_weight=penalty_weight,
    bound_penalty_weight=bound_penalty_weight,
    maxiter=maxiter,
    popsize=popsize,
    seed=seed,
    max_reflectance=0.7
)

# Display optimization results
print("\nOptimization Results:")
print(f"Success: {optimization_result.success}")
print(f"Message: {optimization_result.message}")
print(f"Number of Function Evaluations: {optimization_result.nfev}")
final_volume = compute_hull_volume(optimized_vertices)
print(f"Final Volume of Optimized Convex Hull: {final_volume:.4f}")

# Verify that all points are inside the optimized hull
try:
    hull = ConvexHull(optimized_vertices)
    hull_vertices = optimized_vertices[hull.vertices]
    inside = points_inside_hull(edgemembers, hull_vertices)
    num_inside = np.sum(inside)
    print(f"Number of Original Points Inside the Optimized Hull: {num_inside} / {len(edgemembers)}")
except Exception as e:
    print("Failed to verify point enclosure:", e)

# Plot the results
#plot_convex_hull(edgemembers, optimized_vertices, title='Minimal Volume Convex Hull')

    

# Use the inverse PCA transformation to get the endmembers in the original space
S = pca.inverse_transform(hull_vertices)
# Pre-allocate array with np.full
endmembers = np.full((S.shape[0], sampled_data.shape[1]), np.nan, dtype=np.float32)
endmembers[:, valid_bands] = S

#plot_endmembers(wavelengths, endmembers, title='Optimized Endmembers')



# Save the PCA, the optimized vertices, and the endmembers
joblib.dump(pca, 'pca.joblib')
np.save('optimized_vertices.npy', optimized_vertices)
np.save('endmembers.npy', endmembers)
np.save('edgemembers.npy', edgemembers)



