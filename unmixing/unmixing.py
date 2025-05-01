import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull


def remove_outliers(data, contamination=0.05, random_state=42):
    """Remove outliers using IsolationForest."""
    clf = IsolationForest(contamination=contamination, random_state=random_state)
    prediction = clf.fit_predict(data)
    return data[prediction == 1, :]
                
def smacc(spectra, min_endmembers=None, max_residual_norm=float('Inf')):
    '''Returns SMACC decomposition (H = F * S + R) matrices for an image or
    array of spectra.

    Let `H` be matrix of shape NxB, where B is number of bands, and N number of
    spectra, then if `spectra` is of the same shape, `H` will be equal to `spectra`.
    Otherwise, `spectra` is assumed to be 3D spectral image, and it is reshaped
    to match shape of `H`.

    Arguments:

        `spectra` (ndarray):

            Image data for which to calculate SMACC decomposition matrices.

        `min_endmembers` (int):

            Minimal number of endmembers to find. Defaults to rank of `H`,
            computed numerically with `numpy.linalg.matrix_rank`.

        `max_residual_norm`:

            Maximum value of residual vectors' norms. Algorithm will keep finding
            new endmembers until max value of residual norms is less than this
            argument. Defaults to float('Inf')

    Returns:
        3 matrices, S, F and R, such that H = F * S + R (but it might not always hold).
        F is matrix of expansion coefficients of shape N x num_endmembers.
        All values of F are equal to, or greater than zero.
        S is matrix of endmember spectra, extreme vectors, of shape num_endmembers x B.
        R is matrix of residuals of same shape as H (N x B).

        If values of H are large (few tousands), H = F * S + R, might not hold,
        because of numeric errors. It is advisable to scale numbers, by dividing
        by 10000, for example. Depending on how accurate you want it to be,
        you can check if H is really strictly equal to F * S + R,
        and adjust R: R = H - np.matmul(F, S).

    References:

        John H. Gruninger, Anthony J. Ratkowski, and Michael L. Hoke "The sequential
        maximum angle convex cone (SMACC) endmember model", Proc. SPIE 5425, Algorithms
        and Technologies for Multispectral, Hyperspectral, and Ultraspectral Imagery X,
        (12 August 2004); https://doi.org/10.1117/12.543794
    '''    
    # Flatten 3D spectral image to 2D if necessary
    H = spectra.reshape((-1, spectra.shape[2])) if spectra.ndim == 3 else spectra
    
    R = H.copy()
    Fs = []
    q = []  # Indices of selected vectors in S
    
    # Set default minimum endmembers
    if min_endmembers is None:
        min_endmembers = np.linalg.matrix_rank(H)

    residual_norms = np.linalg.norm(H, axis=1)
    current_max_residual_norm = np.max(residual_norms)
    
    if max_residual_norm is None:
        max_residual_norm = current_max_residual_norm / min_endmembers

    while len(q) < min_endmembers or current_max_residual_norm > max_residual_norm:
        idx = np.argmax(residual_norms)
        q.append(idx)
        
        # Calculate projection coefficients
        w = R[idx]
        wt = w / np.dot(w, w)
        On = np.dot(R, wt)
        alpha = np.ones_like(On)
        
        # Adjust alphas for oblique projection
        for k in range(len(Fs)):
            t = On * Fs[k][idx]
            t[t == 0.0] = 1e-10
            alpha = np.minimum(Fs[k] / t, alpha, where=t != 0.0)
        
        # Clip negative projection coefficients
        alpha[On <= 0.0] = 0.0
        alpha[idx] = 1.0
        
        # Calculate oblique projection coefficients
        Fn = alpha * On.clip(min=0.0)
        R -= np.outer(Fn, w)
        
        # Update existing factors
        for k in range(len(Fs)):
            Fs[k] -= Fs[k][idx] * Fn
            Fs[k] = Fs[k].clip(min=0.0)
        
        Fs.append(Fn)
        residual_norms = np.linalg.norm(R, axis=1)
        current_max_residual_norm = np.max(residual_norms)
        
        print(f'Found {len(q)} endmembers, current max residual norm is {current_max_residual_norm:.4f}\r', end='')

    # Correction as suggested in the SMACC paper.
    for k, s in enumerate(q):
        Fs[k][q] = 0.0
        Fs[k][s] = 1.0

    F = np.column_stack(Fs).T
    S = H[q]

    return S, F, R

def qhull(data,n_components=3):
    """
    Finds the edges of the data simplex using QHULL to find the endmembers.
    
    Parameters:
    data (numpy.ndarray): The input data, expected to be 2D (pixels x bands).
    subsize (int): The maximum number of pixels to use for the analysis.

    Returns:
    realsig (numpy.ndarray): The real signatures from the data.
    """


    # Perform PCA transformation
    pca = PCA(n_components=n_components)
    pc = pca.fit_transform(data)
    
    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f'Cumulative PCA Explained Variance = {cumulative_explained_variance}')
    
    # Use Qhull via ConvexHull to find the extreme points
    # Fx print extreme points (vertices) of convex hulls
    # W0.1 min distance above plane for outside points to approximate the convex hull
    # QbB  - scale the input to fit the unit cube
    hull = ConvexHull(pc, qhull_options='Fx W0.5 QbB')

    # Extract the real signatures
    realsig = data[hull.vertices]

    return realsig

def nfindr(srData,n_components=0.999):
    """
    Extract endmembers from spectral data using the NFINDR algorithm.
    
    Parameters:
    - srData (ndarray): Input spectral reflectance data of shape (bands, samples).
    - nSubsets (int, optional): Number of subsets for subsampling. Defaults to None.
    
    Returns:
    - endmembers (ndarray): Detected endmember spectra.
    """

    # Compute the PCA
    pca = PCA(n_components=n_components)
    pcaData = pca.fit_transform(srData)
    
    # Calculate cumulative explained variance
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    print(f'Cumulative PCA Explained Variance = {cumulative_explained_variance}')
    
    # First two endmembers are the extremes of PCA component 1
    sub = np.array([np.argmax(pcaData[:, 0]), np.argmin(pcaData[:, 0])])

    # Loop to find remaining endmembers
    for nEdge in range(3, pcaData.shape[1] + 1):
        dmax = 0
        endMatrix = np.ones(nEdge)

        for i in range(nEdge - 1):
            endMatrix = np.vstack((endMatrix, np.append(pcaData[sub, i], 0)))

        # Test every pixel to find new endmember
        for i in range(len(pcaData)):
            endMatrix[1:nEdge, nEdge - 1] = pcaData[i, 0:nEdge - 1]
            
            # Calculate volume of the simplex
            volume = np.abs(np.linalg.det(endMatrix))
            
            if volume > dmax and i not in sub:
                dmax = volume
                subn = i
        
        print(f"Completed {nEdge} search\r", end='')
        sub = np.append(sub, subn)

    # Extract the endmembers in actual reflectance
    endmembers = srData[sub]
    return endmembers



import time
from cvxopt import matrix, solvers

class FCLSU:
    def __init__(self):
        pass

    def __repr__(self):
        msg = f"{self.__class__.__name__}"
        return msg

    @staticmethod
    def _numpy_None_vstack(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.vstack([A1, A2])

    @staticmethod
    def _numpy_None_concatenate(A1, A2):
        if A1 is None:
            return A2
        else:
            return np.concatenate([A1, A2])

    @staticmethod
    def _numpy_to_cvxopt_matrix(A):
        A = np.array(A, dtype=np.float64)
        if A.ndim == 1:
            return matrix(A, (A.shape[0], 1), "d")
        else:
            return matrix(A, A.shape, "d")

    def solve(self, Y, E):
        """
        Performs fully constrained least squares of each pixel in M
        using the endmember signatures of U. Fully constrained least squares
        is least squares with the abundance sum-to-one constraint (ASC) and the
        abundance nonnegative constraint (ANC).
        Parameters:
            Y: `numpy array`
                2D data matrix (L x N).
            E: `numpy array`
                2D matrix of endmembers (L x p).
        Returns:
            X: `numpy array`
                2D abundance maps (p x N). d
        References:
            Daniel Heinz, Chein-I Chang, and Mark L.G. Fully Constrained
            Least-Squares Based Linear Unmixing. Althouse. IEEE. 1999.
        Notes:
            Three sources have been useful to build the algorithm:
                * The function hyperFclsMatlab, part of the Matlab Hyperspectral
                Toolbox of Isaac Gerg.
                * The Matlab (tm) help on lsqlin.
                * And the Python implementation of lsqlin by Valera Vishnevskiy, click:
                http://maggotroot.blogspot.ca/2013/11/constrained-linear-least-squares-in.html
                , it's great code.
        """
        tic = time.time()
        assert len(Y.shape) == 2
        assert len(E.shape) == 2

        L1, N = Y.shape
        L2, p = E.shape

        assert L1 == L2

        # Reshape to match implementation
        M = np.copy(Y.T)
        U = np.copy(E.T)

        solvers.options["show_progress"] = False

        U = U.astype(np.double)

        C = self._numpy_to_cvxopt_matrix(U.T)
        Q = C.T * C

        lb_A = -np.eye(p)
        lb = np.repeat(0, p)
        A = self._numpy_None_vstack(None, lb_A)
        b = self._numpy_None_concatenate(None, -lb)
        A = self._numpy_to_cvxopt_matrix(A)
        b = self._numpy_to_cvxopt_matrix(b)

        Aeq = self._numpy_to_cvxopt_matrix(np.ones((1, p)))
        beq = self._numpy_to_cvxopt_matrix(np.ones(1))

        M = np.array(M, dtype=np.float64)
        M = M.astype(np.double)
        X = np.zeros((N, p), dtype=np.float32)
        for n1 in range(N):
            d = matrix(M[n1], (L1, 1), "d")
            q = -d.T * C
    
            sol = solvers.qp(Q, q.T, A, b, Aeq, beq, None, None)["x"]
            X[n1] = np.array(sol).squeeze()
        tac = time.time()
        print(f"{self} took {tac - tic:.2f}s")
        return X.T
