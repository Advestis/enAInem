"""
Created on Fri Jan 17 10:10:00 2025
Version: 1.3.0

@authors: Paul Fogel, Christophe Geissler & George Luta
@E-mail: paul.fogel@forvismazars.com
@Github: https://github.com/Advestis/enAInem/tree/main

The class EnAInem is an extension of the class NMF from scikit-learn:
  - Tensors of any order (Fast HALS)
  - Heterogeneous views (using the Integrated Sources Model)
  - Random Completions (using the Integrated Sources Model)
  - Robust approach (using the Target Polish approach)
Note: The loss criterion is restricted to Frobenius
Acknowledgment: With Advestis part of ForvisMazars support
License: MIT

"""

import warnings
from numbers import Integral, Real
import os
from typing import Tuple, Union, Dict, List
import concurrent.futures
import copy
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import khatri_rao
from scipy.sparse.linalg import svds
from scipy.stats import skew
from scipy.stats import halfnorm
from functools import reduce

from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_array, check_random_state, gen_batches, metadata_routing
from sklearn.utils._param_validation import (
    Interval,
    StrOptions,
    validate_params,
)
# from sklearn.utils.deprecation import _deprecate_Xt_in_inverse_transform
from sklearn.utils.extmath import randomized_svd, safe_sparse_dot, squared_norm
from sklearn.utils.validation import (
    # check_is_fitted,
    check_non_negative,
    # validate_data,
)
from sklearn.decomposition._cdnmf_fast import _update_cdnmf_fast
from sklearn.decomposition._nmf import norm, _BaseNMF, _check_init, _special_sparse_dot, _update_coordinate_descent

EPSILON = np.finfo(np.float32).eps

class ConvergenceWarning(UserWarning):
    """Custom warning to capture convergence problems"""

class OptimizationWarning(UserWarning):
    """Custom warning to capture optimization problems"""

# Define a custom showwarning function
def custom_showwarning(message, category, filename, lineno, file=None, line=None):
    print(f"{category.__name__}: {message}")

# Set the custom showwarning function
warnings.showwarning = custom_showwarning

def _svd(
    X: NDArray,
    k: int,
    use_randomized_svd: bool,
    random_state: Union[int, None],
) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
    """Apply full or truncated SVD.

    Parameters
    ----------
    X:          (nobs, ndim) data matrix, missing values denoted by NaN/Inf.
    k:          number of singular values/vectors to find (default: k=ndim).

    Returns
    -------
    U, s, V:   singular values and vectors (see np.linalg.svd and
                scipy.sparse.linalg.svds for details).
    """
    if k < min(X.shape):
        if use_randomized_svd:
            U, s, Vt = randomized_svd(X, k, random_state=random_state)  # type: ignore
        else:
            U, s, Vt = svds(X, k, random_state=random_state)  # type: ignore

    else:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        U = U[:, :k]
        s = s[:k]
        Vt = Vt[:k, :]

    # Ensure a majority of positive elements in Vt
    if np.median(Vt) < 0:
        U, Vt = -U, -Vt

    if k == 1:
        U, Vt = U[:, 0], Vt[0, :]

    return (U, s, Vt.T)

def _nndsvd(
    X: NDArray,
    n_components: int,
    use_randomized_svd: bool,
    dim_order: list[int],
    random_state: Union[int, None],
    eps=1e-6,
) -> list[NDArray]:
    """Generalize NNDSVD for NTF initialization.

    Returns
    ----------
        arrays (list[NDArray]): Basis vectors in each dimension of the tensor.

    Approach
    --------
        step 1:
        Initialize nndsvd on a tensor of order 2 (matrix), then perform NMF on this tensor.
        The tensor of order 2 is basically an unfolding of the original tensor by
        choosing the axis that has the largest dimension.
        Step 2:
        component by component, perform svd with rank 1, to sequentially “unravel” the other
        dimensions, in descending order of the dimensions.
    """
    n_dims = X.ndim
    dim_restore = np.zeros(n_dims).astype(int)
    dim_restore[dim_order] = range(n_dims)
    Z_shape = [X.shape[dim] for dim in dim_order]
    Z = X.copy()
    Z = np.transpose(Z, axes=dim_order)
    B = [np.zeros((Z_shape[dim], n_components)) for dim in range(len(Z_shape))]
 
    # Step 1
    # Truncated SVD of Z
    U, S, V = _svd(Z.reshape(Z_shape[0], -1), n_components, use_randomized_svd, random_state)
    if n_components == 1:
        U = U[:, np.newaxis]
        V = V[:, np.newaxis]

    # nndsvd on U
    U_pos, V_pos = np.maximum(U, 0), np.maximum(V, 0)
    U_neg, V_neg = U_pos - U, V_pos - V

    for j in range(n_components):
        norm_u_pos, norm_v_pos = norm(U_pos[:, j]), norm(V_pos[:, j])
        norm_u_neg, norm_v_neg = norm(U_neg[:, j]), norm(V_neg[:, j])
        term1, term2 = norm_u_pos * norm_v_pos, norm_u_neg * norm_v_neg
        if term1 > term2:
            sigma = np.sqrt(term1 * S[j])
            U[:, j] = (U_pos[:, j] / norm_u_pos) * sigma
            V[:, j] = (V_pos[:, j] / norm_v_pos) * sigma
        else:
            sigma = np.sqrt(term2 * S[j])
            U[:, j] = (U_neg[:, j] / norm_u_neg) * sigma
            V[:, j] = (V_neg[:, j] / norm_v_neg) * sigma

        B[0] = np.ascontiguousarray(U)
        B[0][B[0] < eps] = 0

    # Step 2 on V
    for dim in range(1, len(Z_shape) - 1):
        shape_product = np.prod(Z_shape[dim + 1:])
        svd_results = [_svd(V[:, j].reshape(Z_shape[dim], shape_product), 1, use_randomized_svd, random_state) 
                       for j in range(n_components)]
        B_slices, S_values, V_slices = zip(*svd_results)
        Sigma = np.sqrt(np.column_stack(S_values))
        B[dim] = np.ascontiguousarray(
            np.maximum(np.column_stack(B_slices), 0) * Sigma
        )
        B[dim][B[dim] < eps] = 0
        V = np.column_stack(V_slices) * Sigma

    B[len(Z_shape) - 1] = np.ascontiguousarray(np.maximum(V, 0))
    return [B[dim] for dim in dim_restore]

def _initialize_ntf(
    X: NDArray,
    n_components: int,
    dim_order: Union[list[int], None],
    init: str,
    use_randomized_svd: bool,
    random_state: Union[int, None]=None,
) -> list[NDArray]:
    """Algorithms for NMF initialization.

    Computes an initial guess for the non-negative
    rank k matrix approximation for X: X = WH.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        The data matrix to be decomposed.

    n_components : int
        The number of components desired in the approximation.

    init :  {'random', 'nndsvd', 'custom'}, default='nndsvd'
        Method used to initialize the procedure.
        Valid options:

        - 'random': non-negative random matrices, scaled with:
            sqrt(X.mean() / n_components).

        - 'nndsvd': Nonnegative Double Singular Value Decomposition (NNDSVD)
            initialization (better for sparseness)

        - 'custom': use custom list of arrays B.
    
    dim_order : int or None, default=None
        The order of dimensions to be used in _nndsvd
        By default, dimensions are sorted by decreasing shape.

    random_state : int, RandomState instance or None, default=None
        Used when ``init`` == 'nndsvdar' or 'random'. Pass an int for
        reproducible results across multiple function calls.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    B : List of NDarrays initializing the factorization.

    References
    ----------
    C. Boutsidis, E. Gallopoulos: SVD based initialization: A head start for
    nonnegative matrix factorization - Pattern Recognition, 2008
    http://tinyurl.com/nndsvd

    The generalization to tensors is straightfoward and described in _nndsvd
    """
    
    n_dims = X.ndim
    check_non_negative(X, "NMF initialization")

    if dim_order is None:
        # order dimensions by decreasing shape
        dim_order = np.argsort(X.shape)[::-1]
    else:
        if len(dim_order) != n_dims or sorted(dim_order) != list(range(n_dims)):
            raise ValueError(
                f"dim_order: {dim_order} misdefined."
            )
    Z_shape = [X.shape[dim] for dim in dim_order]
    
    n_samples, n_features = Z_shape[0], reduce(lambda x, y: x*y, Z_shape[1:])

    if (
        init is not None
        and init != "random"
        and init != "custom"
        and n_components > min(n_samples, n_features)
    ):
        raise ValueError(
            "init = '{}' can only be used when "
            "n_components <= min(n_samples, n_features)".format(init)
        )

    if init is None:
        if n_components <= min(n_samples, n_features):
            init = "nndsvd"
        else:
            init = "random"

    if init == "random":
        rng = check_random_state(random_state)
        avg = (np.nanmean(X) / n_components) ** (1 / X.ndim)
        B = []
        for dim in range(n_dims):
            B.append(
            avg * rng.standard_normal(size=(X.shape[dim], n_components)).astype(
                X.dtype, copy=False
            )) 
        return [np.abs(B[dim], out=B[dim]) for dim in range(X.ndim)]
    else:
        if init == "custom":
            raise ValueError(
                "When init=='custom', B list must be provided. Set "
                "init='nndsvd' or 'random' to initialize B."
            )
        return _nndsvd(X, n_components, use_randomized_svd, dim_order, random_state)

def _generate_tensor(
        B: list[NDArray]
    ) -> NDArray:
    """Calculate the sum of outer products over all components.

    Parameters
    ----------
        B (list[NDArray]): Basis vectors in each dimension of the tensor.

    Returns
    -------
        NDArray: The result of the sum(outer_product by component). Same dimension as X
    """
    X_shape = [B[dim].shape[0] for dim in range(len(B))]
    result = reduce(khatri_rao, B)
    return np.sum(result, axis=1).reshape(X_shape)

def _my_update_coordinate_descent(X, W, XHt, HHt, l1_reg, l2_reg, shuffle, random_state):
    """Helper function for _fit_coordinate_descent.

    Update W to minimize the objective function, iterating once over all
    coordinates. By symmetry, to update H, one can call
    _update_coordinate_descent(X.T, Ht, W, ...).
    
    Customized for enAInem: arguments XHt and HHt are calculated in _fit_coordinate_descent.
    """
    n_components = XHt.shape[1]

    # HHt = np.dot(Ht.T, Ht)
    # XHt = safe_sparse_dot(X, Ht)

    # L2 regularization corresponds to increase of the diagonal of HHt
    if l2_reg != 0.0:
        # adds l2_reg only on the diagonal
        HHt.flat[:: n_components + 1] += l2_reg
    # L1 regularization corresponds to decrease of each element of XHt
    if l1_reg != 0.0:
        XHt -= l1_reg

    if shuffle:
        permutation = random_state.permutation(n_components)
    else:
        permutation = np.arange(n_components)
    # The following seems to be required on 64-bit Windows w/ Python 3.5.
    permutation = np.asarray(permutation, dtype=np.intp)
    return _update_cdnmf_fast(W, HHt, XHt, permutation)

def _update_D(X, B, target_polish="CIM"):
    if target_polish == "CIM":
        E_square = np.square(X - _generate_tensor(B))
        twice_sigma_square = np.mean(E_square)
        return np.exp(- E_square / twice_sigma_square)
    elif target_polish == "Huber":
        abs_E = np.abs(X - _generate_tensor(B))
        abs_E[abs_E < EPSILON] = EPSILON
        c = np.median(abs_E)
        return np.where(abs_E < c, 1, c / abs_E)
    elif target_polish == "L1":
        # `eps` cannot be a too small value like np.finfo(np.float32).eps
        eps = X.var() / B[0].shape[1]
        D = 1 / (np.sqrt(np.square(X - _generate_tensor(B))) + eps ** 2)
        return D / D.max()
    elif target_polish == "L21":
        D = 1 / np.sqrt(np.sum(np.square(X - _generate_tensor(B)), axis=0))
        D = np.repeat(D[np.newaxis, :], X.shape[0], axis=0)
        return D / D.max()

def _ensure_minimum(matrix, epsilon):
    matrix[matrix < epsilon] = epsilon
    return matrix

def _fit_coordinate_descent(
    X: NDArray,
    B: list[NDArray],
    update: list[bool],
    l1_reg: NDArray,
    l2_reg: NDArray,
    tol: float,
    max_iter: int,
    random_state: Union[int, None],
    verbose: int,
    target_polish: Union[str, None],
    target_polish_fraction: float, 
    target_polish_threshold: float,
) -> Tuple[list[NDArray], int, float]:
    """Compute Non-negative Matrix Factorization (NMF) with Coordinate Descent.

    The objective function is minimized with an alternating minimization along the 
    dimensions of B. Each minimization is done with a cyclic Coordinate Descent.
    This function calls a slight modification of the function _update_coordinate_descent
    found in module scikit-learn _nmf and whenever possible uses the same notations.

    Parameters
    ----------
    X : NDArray-like of shape
        Constant matrix.

    B : List of NDArrays
        Initial guess for the solution.

    update : list of booleans, default=True
        update[dim] set to True  B[dim] be estimated from initial guesses.

    l1_reg : NDArray, default= NDArray of 0's.
        L1 regularization parameters for B.

    l2_reg : NDArray, default= NDArray of 0's.
        L2 regularization parameters for B.
  
    tol: float, default=1.e-4
        Tolerance of the stopping condition.
    
    max_iter: int, default=200
        Maximum number of iterations before timing out.

    random_state: , RandomState instance or None, default=0
        Used when ``init`` ==  'random' and random completions. Pass an int 
        for reproducible results across multiple function calls.
   
    verbose: int,  default=0
        The verbosity level.

    target_polish: Literal['CIM', 'Huber', 'L1', 'L21] | None, default='CIM'
        If not None: multiplicative coefficient applied to the original target

    Returns
    -------
    B : List of NDAarrays
        Solution to the non-negative least squares problem.

    n_iter : int
        The number of iterations done by the algorithm.

    scaled_violation : float
        The violation at convergence.

    relative_error: float
        The relative error at convergence.

    References
    ----------
    .. [1] :doi:`"Fast local algorithms for large scale nonnegative matrix and tensor
    factorizations" <10.1587/transfun.E92.A.708>`
    Cichocki, Andrzej, and P. H. A. N. Anh-Huy. IEICE transactions on fundamentals
    of electronics, communications and computer sciences 92.3: 708-721, 2009.
     .. [2] :doi:`"The Target Polish: A New Approach to Outlier-Resistant Non-Negative
     Matrix and Tensor Factorization" https://arxiv.org/abs/2507.10484`
     Fogel, Paul, Geissler, Christophe, and Luta, George. arXiv, Computer Science,
     Machine Learning.
    """
    n_dims = X.ndim
    
    #  so b arrays are in C order in memory
    B = [check_array(b, order="C") for b in B]
    
    # Normalize components prior to updating loop
    n_components = B[0].shape[1]

    avg = (np.nanmean(X) / n_components) ** (1 / X.ndim)
    
    for dim in range(n_dims-1):
        temp = _ensure_minimum(np.mean(B[dim], axis=0), EPSILON) / avg
        B[dim] /= temp    
        B[-1] *= temp
    
    B[-1] *= np.mean(X) / max(np.mean(_generate_tensor(B)), EPSILON)
    # Parameters required by sklearn update_coordinate_descent (but not used for now)
    rng = check_random_state(random_state)
    shuffle = False

    # Note: it is very important to ensure that all entries in B[dim].T @ B[dim] are >= EPSILON 
    # to prevent the situation where the dot product yields some zero entries in some dimension,
    # which would cancel the corresponding buffer entries due to chained multiplication.

    HHt_buffer = reduce(
        lambda x, y: _ensure_minimum(x * y, EPSILON),
        [B[dim].T @ B[dim] for dim in range(n_dims)]
    )
    
    if target_polish is not None:
        avg = np.mean(X)
        # Calculate the step size to calculate the target change on size/1000 values to save time
        num_elements = X.size
        step_size = int(np.ceil((num_elements * target_polish_fraction)**(1/X.ndim)))
    else:
        step_size = 1

    slice_obj = tuple(slice(None, None, step_size) for _ in range(X.ndim))
    
    DX = X.copy()
    DX_0_slice = DX[slice_obj].copy()

    for n_iter in range(1, max_iter+1):
        violation = 0
        if target_polish is not None:
            B_slice = list(B[dim][slice_obj[dim]] for dim in range(X.ndim))
            D_slice = _update_D(X[slice_obj], B_slice, target_polish=target_polish)
            DX_slice = avg * (1 - D_slice) + D_slice * X[slice_obj]
            target_change = norm(DX_slice - DX_0_slice) / max(norm(DX_0_slice), EPSILON)
            if target_change > target_polish_threshold:
                DX_0_slice = DX_slice.copy()
                D = _update_D(X, B, target_polish=target_polish)
                DX = avg * (1 - D) + D * X
            
        for dim in range(n_dims):
            if update[dim]:
                if n_dims > 2:
                    indices = [dim] + [i for i in range(n_dims) if i != dim]
                    X_moved = np.moveaxis(DX, source=indices, destination=list(range(n_dims)))
                    X_dim = np.reshape(X_moved, (X.shape[dim], -1))
                else:
                    X_dim = DX if dim == 0 else DX.T
                indices = [i for i in range(n_dims) if i != dim]
                XHt = safe_sparse_dot(
                    X_dim,
                    reduce(khatri_rao, [B[i] for i in indices])
                )
                HHt = HHt_buffer / _ensure_minimum(B[dim].T @ B[dim], EPSILON)
                # HHt is copied since it may be modified by __update_coordinate_descent
                HHt_copy = HHt.copy() if l2_reg[dim] != 0.0 else HHt
                violation += _my_update_coordinate_descent(
                    X_dim, B[dim], XHt, HHt_copy, l1_reg[dim], l2_reg[dim], shuffle, rng
                )
                np.copyto(HHt_buffer, HHt * _ensure_minimum(B[dim].T @ B[dim], EPSILON))

                # Alternative update using unmodified sklearn update is sub-optimal
                #     indices = [i for i in range(n_dims) if i != dim]
                #     Ht = reduce(
                #         khatri_rao, 
                #         [B[i] for i in indices]
                #     )
                #     violation += _update_coordinate_descent(
                #         X_dim, B[dim], Ht, l1_reg[dim], l2_reg[dim], shuffle, rng
                #     )

        if n_iter == 1:
            violation_init = violation

        if violation_init == 0:
            break

        scaled_violation = violation / violation_init
        if verbose == 2:
            print(f"#{n_iter} violation: {scaled_violation}")
        if scaled_violation <= tol:
            if verbose == 2:
                print(f"Converged at iteration {n_iter + 1}.")
            break
        elif n_iter == max_iter and tol > 0:
            warnings.warn(
                (
                    f"Maximum number of iterations {max_iter} reached."
                    " Increase it to improve convergence."
                ),
                ConvergenceWarning,
            )

    if target_polish is not None:
        # Complete approximation with weighted mult rules to adjust real X
        # First update the weights obtained at the end of the Target Polish iterations 
        # and use them fixed in further weighted mult rules.
        D = _update_D(X, B, target_polish=target_polish)
        error_slice = [None, None]
       
        for n_iter in range(1, max_iter + 1):
            for dim in range(n_dims):
                if update[dim]:
                    if n_dims > 2:
                        indices = [dim] + [i for i in range(n_dims) if i != dim]
                        X_moved = np.moveaxis(X, source=indices, destination=list(range(n_dims)))
                        X_dim = np.reshape(X_moved, (X.shape[dim], -1))
                        D_moved = np.moveaxis(D, source=indices, destination=list(range(n_dims)))
                        D_dim = np.reshape(D_moved, (X.shape[dim], -1))
                        Ht = reduce(
                            khatri_rao, 
                            [B[i] for i in indices[1:]]
                        )
                    else:
                        X_dim = X if dim == 0 else X.T
                        D_dim = D if dim == 0 else D.T
                        Ht = B[(dim+1)%2]
                    
                    H = Ht.T
                    # update D
                    denominator= _ensure_minimum((D_dim * (B[dim] @ H)) @ Ht, EPSILON)
                    B[dim] = B[dim] * ((D_dim * X_dim) @ Ht) / denominator

            # Assess error change on random slices
            error_slice[-2] = error_slice[-1]
            B_slice = list(B[dim][slice_obj[dim]] for dim in range(X.ndim))
            X_hat_slice = _generate_tensor(B_slice)
            error_slice[-1] = norm(X[slice_obj] - X_hat_slice)
            if n_iter == 1:
                relative_error_slice = np.inf
            else:
                relative_error_slice = np.abs(error_slice[-1] - error_slice[-2]) / error_slice[-2]
       
            if verbose == 2:
                print(f"#{n_iter} relative error (slices): {relative_error_slice}")
            if relative_error_slice <= tol:
                if verbose == 2:
                    print(f"Converged at iteration {n_iter + 1}.")
                break
            elif n_iter == max_iter and tol > 0:
                warnings.warn(
                    (
                        f"Maximum number of iterations {max_iter} reached."
                        " Increase it to improve convergence."
                    ),
                    ConvergenceWarning,
                )

    # Calculate relative norm of residual tensor
    X_hat = _generate_tensor(B)
    E = X - X_hat
    relative_error = norm(E) / max(norm(X), EPSILON)
    return (B, n_iter, scaled_violation, relative_error)

def _ic_complete_with_random_values(X, rng):
    """Complete missing values with values drawn at random from the array."""

    # Flatten the array
    flat_X = X.flatten().astype(float)
    
    # Identify non-missing (non-NaN) values
    non_nan_values = flat_X[~np.isnan(flat_X)]
    
    # Identify indices of missing (NaN) values
    nan_indices = np.where(np.isnan(flat_X))[0]
    
    # Randomly select non-missing values to replace NaNs
    replacement_values = rng.choice(non_nan_values, size=nan_indices.size, replace=True)
    
    # Replace NaNs with the selected non-missing values
    flat_X[nan_indices] = replacement_values
    
    # Reshape the array back to its original shape
    X_comp = flat_X.reshape(X.shape)
    return X_comp

def _ic_parallel_fit(args):
    """Parallelize multiple fit_transform's from within fit_irc."""
    cls, completion, X_comp, B_init, update, alpha, l1_ratio = args
    res = cls.fit_transform(
        X_comp,
        B=B_init,
        update=update,
        alpha=alpha,
        l1_ratio=l1_ratio,
    )
    B = res['B']
    relative_error = res["relative_error"]
    if cls.verbose >= 1:
        print(f"relative error = {relative_error}")
    return (completion, B, relative_error)

def _ic_parallel_fit_wrapper(list_args, optimal_workers):
    with concurrent.futures.ProcessPoolExecutor(max_workers=optimal_workers) as executor:
        results = list(executor.map(_ic_parallel_fit, list_args))

    return results

def _fit_integrate_completions(
        cls: object,
        X: Union[list[NDArray], NDArray],
        B: Union[list[NDArray], None],
        update: Union[list[bool], None],
        alpha: Union[list[float], None],
        l1_ratio: float = 1.0,
    ) -> Tuple[list[NDArray], float]:
    """Integrate Random Completions (IRC) to handle missing values.

    Parameters
    ----------
    X : NDArray
        Constant tensor.

    B : list[NDArray]
        Initial guess for the solution.

    update : list of booleans, default=True
        update[dim] set to True  B[dim] be estimated from initial guesses.
    
    alpha: float, constant that multiplies the regularization terms in each dimension.
        Set the nth element to zero to have no regularization
        on the nth dimension.

    l1_ratio: float, the regularization mixing parameter, with 0 <= l1_ratio <= 1.
        l1_ratio = 0: the penalty is an element wise L2 penalty (aka Frobenius Norm).
        l1_ratio = 1: it is an element wise L1 penalty.
        0 < l1_ratio < 1: the penalty is a combination of L1 and L2.
    
    Returns
    -------

    Approach
    --------
        1- Perform random completions
        2- Apply fast-hals on each completion
        3- Integrate solutions using ism, ntf or mean
    """
    verbose = cls.verbose
    n_dims = X.ndim
    num_cores = os.cpu_count()
    optimal_workers = min(32, num_cores + 0)

    # Loop to factorize completed tensors
    rng = check_random_state(cls.random_state)
    X_comp_list = []
    X_comp_list.extend(_ic_complete_with_random_values(X, rng) for _ in range(cls.n_completions))

    if not cls.irc_parallel_fit_1:
        B_list = []
        for completion in range(cls.n_completions):
            res = cls.fit_transform(
                X_comp_list[completion],
                B=B,
                update=update, 
                alpha=alpha, 
                l1_ratio=l1_ratio,
            )
            relative_error = res["relative_error"]
            res_B = res['B']
            B_list.append(res_B[0])
            if verbose >= 1:
                print(f"relative error = {relative_error}")
    else:
        warnings.warn("Starting parallel processing (1)...", UserWarning)                                         
        list_args = [(cls, completion, X_comp_list[completion], B, update, alpha, l1_ratio) 
                                for completion in range(cls.n_completions)]
        results = _ic_parallel_fit_wrapper(list_args, optimal_workers)
        B_list = [None] * cls.n_completions
        for completion, res_B, relative_error in results:
            # B_list[completion] = normalize(res_B[0], axis=0, copy=False)
            B_list[completion] = res_B[0]
            
    # Integration
    # Inhibate verbose, dim_order and target_polish during integration
    original_verbose = cls.verbose
    cls.verbose = 0
    original_order_dim = cls.dim_order
    cls.dim_order = None
    original_dynamic_target = cls.target_polish
    cls.target_polish = None
    if cls.integrator == "ism":
        # Run ISM on B_list
        if verbose >= 1:
            print("Applying ism...")
        res = cls.fit_transform(B_list)
        W = res['B'][0]
        relative_error = res["relative_error"]
    elif cls.integrator == "nmf":
        # Alternatively run NMF on concatenated B_list
        W_concat = np.concatenate(B_list, axis=1)
        res = cls.fit_transform(W_concat)
        W = res['B'][0]
        relative_error = res["relative_error"]
    elif cls.integrator == "mean":
        W = sum(B_list[completion] for completion in range(cls.n_completions)) / cls.n_completions

    # Restore original class parameters
    cls.verbose = original_verbose
    cls.dim_order = original_order_dim
    cls.target_polish = original_dynamic_target
    if cls.verbose >= 1 and cls.integrator != "mean":
        print(f"relative error = {relative_error}")
        print(f"{cls.integrator} applied.")

    # Loop this time using fixed ISM W
    original_init_mode = cls.init
    cls.init = "custom"
    B = [np.zeros_like(res_B[dim]) for dim in range(n_dims)]
    avg = (np.nanmean(X) / (W.mean() * cls.n_components))**(1 / (n_dims-1))
    # avg = 1
    B_init = [avg * np.ones_like(res_B[dim]) for dim in range(n_dims)]
    B_init[0] = W
    # Send a copy of B_init to fit_transform to preserve B_init 
    B_init_copy = copy.deepcopy(B_init)
    update_2 = [False] + [update[dim] for dim in range(1, n_dims)]

    if not cls.irc_parallel_fit_2:
        for completion in range(cls.n_completions):
            # Send a copy of B_init to preserve argument to fit_transform
            res = cls.fit_transform(
                X_comp_list[completion], 
                B = B_init_copy,
                update=update_2,
                alpha=alpha,
                l1_ratio=l1_ratio,
            )
            for dim in range(n_dims):
                B[dim] += res['B'][dim]
            relative_error = res["relative_error"]
            if verbose >= 1:
                print(f"completion = {completion}; relative error = {relative_error}")
    else:
        warnings.warn("Starting parallel processing (2)...", UserWarning) 
        # Send a copy of B_init to preserve argument to fit_transform
        list_args = [
            (cls, completion, X_comp_list[completion], B_init_copy, update_2, alpha, l1_ratio) 
            for completion in range(cls.n_completions)
        ]
        results = _ic_parallel_fit_wrapper(list_args, optimal_workers)
        if verbose >= 1:
            print("Finished parallel processing (2).")
        for completion, res_B, relative_error in results:
            for dim in range(n_dims):
                B[dim] += res_B[dim]
            if verbose >= 1:
                print(f"completion = {completion}; relative error = {relative_error}")

    for dim in range(n_dims):
        B[dim] /= cls.n_completions

    cls.init = original_init_mode

    # Calculate relative error (wrt last random completion only)
    X_hat = _generate_tensor(B)
    E = X - X_hat
    relative_error = norm(np.nan_to_num(E)) / norm(np.nan_to_num(X))
    return (B, relative_error)

def _is_integrate_views(
    cls,
    concat: object,
    embed: object,
    B: NDArray,
    view_ind: list[(int, int)],
    update: bool,
) -> Tuple[NDArray]:
    """Core integration function called by _fit_integrate_sources.

    Parameters
    ----------
    concat: object (X_nan_to_0, X_weight and H used)
        concat.X_nan_to_0: NDArray
            Concatenated views with missing values replaced by 0's (used with multiplicative rules).

        concat.X_weight: NDArray
            Weights in concatenated views (1=non-missing entry, 0=missing entry).

        concat.H: NDArray
            Concatenated view-mappings

    B: NDArray
        Initialized metascores.

    embed: object (Q used)
        embed.Q: NDArray
            View loadings in embedding space.

    view_ind: list[(int, int)]
        List of tuplet indexes that delimit views in concatenated format.

    update: boolean
        update B
    
    Returns
    -------
    HHIi: NDArray
        Number of non-negligable values in concatenated features by component.

    B: list[NDArray]
        B[0] contains ISM metascores.
        B[1] contains list of view-mapping NDarrays.

    embed: object
        embed.H: NDArray
            NTF loadings in latent space.

        embed.Q: NDArray
            View loadings in embedding space.

        embed.X: NDArray
            Embedded views.
    """
    n_iter_mult, use_fast_mult_rules, update_embed, sparsity_coeff = (
        cls.n_iter_mult,
        cls.use_fast_mult_rules,
        cls.update_embed,
        cls.sparsity_coeff
    )
    n_views = len(view_ind)
    if embed.X is None:
        # Initialize embedding tensor
        embed.X = np.zeros((B.shape[0], B.shape[1], n_views))
              
    # Extract view-related items
    for view in range(n_views):
        i1 = view_ind[view][0]
        i2 = view_ind[view][1]
        X_view_0, X_view_wgt = concat.X_nan_to_0[:, i1:i2], concat.X_weight[:, i1:i2]
        B_view, H_view = B.copy(), concat.H[i1:i2, :] # Make B persistent within loop

        # Apply multiplicative updates to preserve concat.H sparsity
        for _ in range(0, n_iter_mult):
            # Weighted multiplicative rules handle missing values
            if use_fast_mult_rules:
                # do not update estimated view after concat.H update
                X_view_hat = X_view_wgt * (B_view @ H_view.T)
                numerator = safe_sparse_dot(B_view.T, X_view_0)
                denominator = _ensure_minimum(B_view.T @ X_view_hat, EPSILON)
                H_view *= (numerator / denominator).T
                numerator = safe_sparse_dot(X_view_0, H_view)
                denominator = _ensure_minimum(X_view_hat @ H_view, EPSILON)
                B_view *= numerator / denominator
            else:
                numerator = safe_sparse_dot(B_view.T, X_view_0)
                denominator = _ensure_minimum(B_view.T @ ((B_view @ H_view.T) * X_view_wgt), EPSILON)
                H_view *= (numerator / denominator).T
                numerator = safe_sparse_dot(X_view_0, H_view)
                denominator =_ensure_minimum((X_view_wgt * (B_view @ H_view.T)) @ H_view, EPSILON)
                B_view *= numerator / denominator

        # Normalize B_view by max column and update H_view
        temp = _ensure_minimum(np.linalg.norm(B, ord=1, axis=0), EPSILON)
        # temp = _ensure_minimum(np.max(B, axis=0), EPSILON)
        # temp = 1
        B_view /= temp
        H_view *= temp

        # Generate embedding tensor
        embed.X[:, :, view] = B_view
  
    # Apply NTF with n_components and update components
    if embed.Q is None:
        # First integration: NTF updates preliminary embedding achieved by NMF
        res = cls.fit_transform(
            embed.X
        )
    else:
        # Iterated integration or initialized B: NTF initialized by former B, embed.H & .Q 
        original_init_mode = cls.init
        cls.init = "custom"
        res = cls.fit_transform(
            embed.X,
            B=[B, embed.H, embed.Q],
            update=[update, update_embed, True],
        )
        cls.init = original_init_mode
    
    B, embed.H, embed.Q = res["B"]

    # Update loadings based on concat.H (initialized by multiplicative updates)
    concat.H = concat.H @ embed.H
    HHIi = _is_make_H_sparse(concat, embed, view_ind, sparsity_coeff)

    # Dot product does not allow for in-place operation, send back updated H
    return (HHIi, B, embed)

def _is_make_H_sparse(
        concat: object,
        embed: object, 
        view_ind: list[(int, int)], 
        sparsity_coeff: float,
    ) -> Tuple[NDArray, NDArray]:
    """Calculate HHIi of each H column and generate sparse loadings.

    Parameters
    ----------
    concat: object, uses concat.H only
        View mapping, concatenated.
    
    embed: object (Q used)
        embed.Q: NDArray
            View loadings in embedding space.

    view_ind: list[(int, int)]
        List of tuplet indexes that delimit views in concatenated format.

    sparsity_coeff: float=0.8
        Enhance embed.H sparsity by a multiplicative factor applied to the inverse HHI.

    Returns
    -------
    HHIi: NDArray
        Number of non-negligable values in concatenated features by component.

    Note: H is modified in place to provide a sparse H.

    """
    n_embed = concat.H.shape[1]
    n_views = len(view_ind)
    HHIi = np.ones(n_embed, dtype=np.uint64)
    H_threshold = np.zeros(n_embed)
    if embed.Q is not None:
        for view in range(n_views):
            concat.H[view_ind[view][0]:view_ind[view][1], :] *= embed.Q[view]

    for j in range(0, n_embed):
        # calculate inverse hhi
        if np.max(concat.H[:, j]) > 0:
            HHIi[j] = int(round(np.sum(concat.H[:, j]) ** 2 / np.sum(concat.H[:, j] ** 2)))
        # sort the dataframe by score in descending order
        H_threshold[j] = (
            np.sort(concat.H[:, j], axis=0)[::-1][int(HHIi[j] - 1)] * sparsity_coeff
        )
        concat.H[concat.H < H_threshold[None, :]] = 0
    return HHIi

def _is_view_indexes(view_widths):
    indexes = []
    start_index = 0

    for width in view_widths:
        end_index = start_index + width
        indexes.append((start_index, end_index))
        start_index = end_index
    
    return indexes

def _is_setup(
        X: list[NDArray], 
        H_mask: Union[list[NDArray], None],
        n_components: int,
    ) -> Tuple[Union[list[NDArray], int, list[(int,int)]]]:
    """Setup of arrays that will be used by _fit_integrate_sources.

    Parameters
    ----------
    X: list(NDArray)
        List of views.

    H_mask: list(NDArray) | None=None
        View-mapping mask (to enforce zero attributes).
  
    Returns
    -------
    concat: object
        concat.X: NDArray
            Concatenated views.

        concat.X_nan_to_0: NDArray
            Concatenated views with missing values replaced by 0's (used with multiplicative rules).

        concat.X_weight: NDArray
            Weights in concatenated views (1=non-missing entry, 0=missing entry).

        concat.H_mask: NDArray
            Concatenated view-masks (initialization).
        
        concat.H: NDArray
            Concatenated view-mappings (initialization).

    embed: object
        Structure containing embedding components H, Q, and embedding tensor X
        (all initialized at None value).

    view_ind: list[(int, int)]
        List of tuplet indexes that delimit views in concatenated format.

    n_views: int
        Number of views.
    """
    concat = Concat()
    embed = Embed()

    concat.X = X[0].copy()
    for V in X[1:]:
        concat.X = np.ascontiguousarray(np.hstack((concat.X, V)))

    view_ind = _is_view_indexes([X[view].shape[1] for view in range(len(X))])
    n_views = len(X)

    # Create X_concat_w with ones and zeros if not_missing/missing entry
    concat.X_weight = np.where(np.isnan(concat.X), 0, 1)

    # Create concat.X_nan_to_0 replacing missing values in concat.X by 0s
    # (to be used with multiplicative rules)
    concat.X_nan_to_0 = concat.X.copy()
    concat.X_nan_to_0[np.isnan(concat.X_nan_to_0)] = 0
    
    if H_mask is not None:
        for view in range(n_views):
            concat.H_mask[view_ind[view][0]:view_ind[view][1], :] = H_mask[view]

    concat.H = np.ones((concat.X.shape[1], n_components))
  
    return (concat, embed, view_ind, n_views)

def _fit_integrate_sources(
    cls,
    X: list[NDArray],
    B: Union[NDArray, None] = None,
    update: bool = True,
    ) -> Tuple[list[NDArray], int, float]:
    """Estimate ISM model.

    Parameters
    ----------
    cls: obj
        Instance of calling class.

    X: List[NDArray]
        List of views.

    B: NDArray
        Initialized metascores.

    update: boolean
        Update metascores.


    Returns
    -------
    H_map: list[NDArray]
        Sparse view mappings.

    HHIi: NDArray
        Number of non-negligable values in concatenated features by component.

    B: list[NDArray]
        B[0] contains ISM metascores.
        B[1] contains list of view-mapping NDarrays.

    embed: object
        embed.H: NDArray
            NTF loadings in latent space.

        embed.Q: NDArray
            NTF view loadings.

        embed.X: NDArray
            Embedded views.

    n_iter: int
        Number of integrations.
    
    relative_error: float
        Model relative error.

    References
    ----------
    Paul Fogel, Christophe Geissler, Franck Augé, Galina Boldina, George Luta.
    Integrated sources model: B new space-learning model for heterogeneous multi-view
    data reduction, visualization, and clustering.
    Artificial Intelligence in Health 2024, 1(3), 89–113.
    https://doi.org/10.36922/aih.3427
    """
    
    verbose, n_components, H_mask, sparsity_coeff, max_iter_int = (
        cls.verbose,
        cls.n_components,
        cls.H_mask,
        cls.sparsity_coeff,
        cls.max_iter_int
    )

    if cls.n_embed is None:
        n_embed = n_components
    else:
        n_embed = cls.n_embed

    if B is not None and n_embed != n_components:
        warnings.warn(
            (
                f"When n_embed (={n_embed}) != n_components (={n_components}), "
                " provided B is ignored.", 
            ),
            RuntimeWarning
        )
        B = None
        update = True
    
    concat, embed, view_ind, n_views = _is_setup(X, H_mask, n_components)

    # Enforce 'nndsvd' init whan calling fit_transform in the context of ism
    original_init = cls.init
    cls.init = 'nndsvd'
    
    if B is None:

        # -------------------------------------------------
        # Perform initial embedding with n_embed components
        # -------------------------------------------------

        original_n_components = cls.n_components
        original_dynamic_target = cls.target_polish
        cls.target_polish = None
        cls.n_components = n_embed
        res = cls.fit_transform(concat.X)
        B = res["B"][0]
        cls.n_components = original_n_components
        cls.target_polish = original_dynamic_target
        concat.H = res["B"][1]
        # temp = 1
        temp = _ensure_minimum(np.linalg.norm(B, ord=1, axis=0), EPSILON)
        # temp = _ensure_minimum(np.max(B, axis=0), EPSILON)
        # temp = _ensure_minimum(np.percentile(B, 95, axis=0), EPSILON)
        B /= temp
        concat.H *= temp
        if np.isclose(temp, EPSILON).any():
            warnings.warn(
                'ISM may produce suboptimal results due to a null component when initialized.',
                OptimizationWarning
            )

        # Initialize view-mapping
        if concat.H_mask is not None:
            concat.H *= concat.H_mask
        
        _ = _is_make_H_sparse(concat, embed, view_ind, sparsity_coeff)
        
        # Embed using scores B found in preliminary NMF
        # Initialize components using NMF/NTF
        HHIi, B, embed = _is_integrate_views(
            cls,
            concat,
            embed,
            B,
            view_ind,
            update,
        )
        relative_error = norm(concat.X - B @ concat.H.T) / norm(concat.X)
        if verbose >= 1:
            print(f"relative error before straightening = {round(relative_error, 2)}")
    else:
        embed.H = np.ones((n_components, n_components))
        embed.Q = np.ones((n_views, n_components))
        HHIi = np.ones(n_components)

    # -------------------------------------------------------------------------
    # Iterate embedding with components subtensor until sparsity becomes stable
    # -------------------------------------------------------------------------

    flag = 0
    n_iter = 0
    if n_embed != n_components:
        # Reset embedding tensor since embedding dimension changes
        embed.X = None
    if verbose == 2:
        print("Straightening...")
    for n_iter in range(1, max_iter_int + 1):
        if verbose == 2:
            print(f"iteration  {n_iter}...")
        HHIi_update_0 = HHIi.copy()
        if n_iter == 1:
            # During first iteration, embed.H is set to identity(n_components)
            # It will be later updated if update_embed is set to True
            embed.H = np.identity(n_components)
            HHIi, B, embed = _is_integrate_views(
                cls,
                concat,
                embed,
                B,
                view_ind,
                update,
            )
        else:
            HHIi, B, embed = _is_integrate_views(
                cls,
                concat,
                embed,
                B,
                view_ind,
                update,
            )
        if (HHIi == HHIi_update_0).all():
            flag += 1
        else:
            flag = 0
        if flag == 3:
            break

    # restore original init parameter
    cls.init = original_init

    # Convert H into mapping list H_map
    H_map = []
    for view in range(n_views):
        H_map.append(concat.H[view_ind[view][0]:view_ind[view][1], :])

    relative_error = norm(np.nan_to_num(concat.X - B @ concat.H.T)) / norm(np.nan_to_num(concat.X))
    if verbose >= 1:
        print(f"relative error after straightening = {round(relative_error, 2)}")
    return (H_map, HHIi, B, embed, n_iter, relative_error)

def _running_in_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            raise ImportError("console")
    except (ImportError, AttributeError):
        return False
    return True

# Helper structures used by _fit_integrate_sources
@dataclass
class Concat:
    X: NDArray = None
    X_nan_to_0: NDArray = None
    X_weight: NDArray = None
    H: NDArray = None
    H_mask = None

@dataclass
class Embed:
    X: NDArray = None
    H: NDArray = None
    Q: NDArray = None

class EnAInem(BaseEstimator):
    """Bundle of NTF and ISM, inherits from sklearn _BaseNMF methods

    Parameters
    ----------
    n_components: int
        Number of components.
    
    init: Literal['random', 'nndsvd', 'custom'], default='nndsvd'
        The method used for the initialization of the NTF decomposition.

    use_randomized_svd: bool, default=False
        use scikit-learn randomized svd. If False, numpy.linalg.svds is used.

    tol: float, default=1.e-4
        Tolerance of the stopping condition.

    max_iter: int, default=200
        Maximum number of iterations before timing out.

    random_state: , RandomState instance or None, default=0
        Used when ``init`` ==  'random' and random completions. Pass an int 
        for reproducible results across multiple function calls.

    verbose: int,  default=0
        The verbosity level.

    dim_order: list, default=None
        The ordering of dimensions to be considered during nndsvd initialization.

    target_polish: Literal['CIM', 'Huber', 'L1', 'L21] | None, default='CIM'
        If not None: multiplicative coefficient applied to the original target

    target_polish_fraction, float, default=1.e-3
        The fraction of the polished target used to assess whether the target
        should be updated.

    target_polish_threshold, float, default=0.05
        Update the target if the relative difference between the current target
        and the new target is greater than the threshold.

    ### Random completions specifics (experimental for NTF)
    force_all_finite: boolean, default=True
        Don't allow missing data and don't apply random completions

    n_completions: int, default=5
        Number of random completions performed by the IRC algorithm.

    integrator: str, default='ism'
        Valid options: 'mean', 'nmf' or 'ism' integrate random completions.

    irc_parallel_fit_1: boolean, default=True
        Parallelize fit_transform prior to ism in random completions
        (inactivated if launched from a workbook).

    irc_parallel_fit_2: boolean, default=False
        Parallelize fit_transform after ism in random completions.
        (inactivated if launched from a workbook).

    ### ISM specifics
    n_embed: int or None, default=None
        Dimension of the embedding space (if None set to the number of components).

    max_iter_int: int, default=20
        Max number of iterations during the straightening process.

    n_iter_mult: int, default=200
        Number of iterations of NMF multiplicative updates during the embedding.

    use_fast_mult_rules: boolean, default=True
        Use common matrix estimate in B and H updates.

    sparsity_coeff: float, default=0.8
        Enhance embed.H sparsity by a multiplicative factor applied to the inverse HHI.

    update_embed: boolean, default=True
        Update or not the NTF factoring matrix embed.H in the embedding space.

    H_mask: list(NDArray) or None, default=None
        View-mapping mask (to enforce zero attributes).
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "init": [StrOptions({"random", "nndsvd", "custom"}), None],
        "use_randomized_svd": ["boolean"],
        "tol": [Interval(Real, 0, None, closed="left")],
        "max_iter": [Interval(Integral, 1, None, closed="left")],
        "random_state": ["random_state"],
        "verbose": ["verbose"],
        "dim_order": [list, None],
        "target_polish": [StrOptions({"CIM", "Huber", "L1", "L21"}), None],
        "target_polish_fraction": [Interval(Real, 0, None, closed="left")],
        "target_polish_threshold": [Interval(Real, 0, None, closed="left")],
        # Random completions specifics
        "force_all_finite": ["boolean"],
        "n_completions": [Interval(Integral, 2, None, closed="left")],
        "integrator": [StrOptions({"mean", "nmf", "ism"})],
        "irc_parallel_fit_1": ["boolean"],
        "irc_parallel_fit_2": ["boolean"],
        # ISM specifics
        "n_embed": [Interval(Integral, 0, None, closed="left"), None],
        "max_iter_int": [Interval(Integral, 0, None, closed="left")],
        "n_iter_mult": [Interval(Integral, 20, None, closed="left")],
        "use_fast_mult_rules": ["boolean"],
        "sparsity_coeff": [Interval(Real, 0, None, closed="left")],
        "update_embed": ["boolean"],
        "H_mask": [list, None],
    }

    def __init__(
        self,
        n_components,
        *,
        init='nndsvd',
        use_randomized_svd=False,
        tol=1e-4,
        max_iter=200,
        random_state=0,
        verbose=0,
        dim_order=None,
        target_polish=None,
        target_polish_fraction=0.001,
        target_polish_threshold=0.05,
        force_all_finite=True,
        n_completions=5,
        integrator="ism",
        irc_parallel_fit_1=True,
        irc_parallel_fit_2=False,
        n_embed=None,
        max_iter_int=20,
        n_iter_mult=200,
        use_fast_mult_rules=True,
        sparsity_coeff=0.8,
        update_embed=True,
        H_mask=None,
    ):
        self.n_components = n_components
        self.init = init
        self.use_randomized_svd = use_randomized_svd
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        self.dim_order = dim_order
        self.target_polish = target_polish
        self.target_polish_fraction = target_polish_fraction
        self.target_polish_threshold = target_polish_threshold
        self.force_all_finite = force_all_finite
        self.n_completions = n_completions
        self.integrator = integrator
        self.irc_parallel_fit_1 = irc_parallel_fit_1
        self.irc_parallel_fit_2 = irc_parallel_fit_2
        self.n_embed = n_embed
        self.max_iter_int = max_iter_int
        self.n_iter_mult = n_iter_mult
        self.use_fast_mult_rules = use_fast_mult_rules
        self.sparsity_coeff = sparsity_coeff
        self.update_embed = update_embed
        self.H_mask = H_mask

        self._validate_params()

        if _running_in_notebook() and (self.irc_parallel_fit_1 or self.irc_parallel_fit_2):
            # Disable parallel operations
            warnings.warn(
                "EnAInem called from Jupyter Notebook. Disabling parallel operations.",
                ResourceWarning
            )
            self.irc_parallel_fit_1 = False
            self.irc_parallel_fit_2 = False
    
    def get_param(self, name: str):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise ValueError(f"Parameter '{name}' does not exist in the model.")

    def set_param(self, name: str, value): 
        if not hasattr(self, name):
            raise ValueError(f"Parameter '{name}' does not exist in the model.") 
        setattr(self, name, value)
        self._validate_params()

    # @_fit_context(prefer_skip_nested_validation=True)
    # (decorator not needed since class parameters already validated in __init__)
    @validate_params(
    {
        "X": [list, "array-like"],
        "B": [list, "array-like", None],
        "update": [list, "boolean", None],
        "alpha": [Interval(Real, 0, 1, closed="both"), None],
        "l1_ratio": [Interval(Real, 0, 1, closed="both"), None],
    },
    prefer_skip_nested_validation=True,
    )

    def fit_transform(
        self,
        X: Union[list[NDArray], NDArray],
        B: Union[list[NDArray], NDArray, None] = None,
        update: Union[list[bool], bool, None] = None,
        alpha: Union[list[float], None] = None,
        l1_ratio: float = 1.0,
    ) -> dict[str, Union[Tuple[NDArray], Tuple[NDArray, list[NDArray]], int, float]]:
        """Apply NTF or ISM transformations depending on the input format.

        Parameters
        ----------
            NTF:
                X : NDArray
                    Constant tensor.

                B : list[NDArray] or None, default=None.
                    Initial guess for the solution.
                    'B' notation denotes 'base' vectors on each dimension.
                    
                update: list[bool] or none, default=None
                    Update B with respect to each dimension of the tensor.

                alpha: List of constant that multiplie the regularization terms
                    in each dimension, or None, default=None.
                    Set the nth element to zero to have no regularization
                    on the nth dimension.

                l1_ratio: The regularization mixing parameter, with 0 <= l1_ratio <= 1, default=1.
                    l1_ratio = 0: the penalty is an element wise L2 penalty (aka Frobenius Norm).
                    l1_ratio = 1: it is an element wise L1 penalty.
                    0 < l1_ratio < 1: the penalty is a combination of L1 and L2.

                Note: Most notations are borrowed from scikit-learn nmf module.

            ISM:
                X: list[NDArray]
                    list of 2D views.

                B: NDArray or None, default=None.
                    Initialize metascores.

                update: boolean or None, default=None
                    Update iniiialized metascores.

        Returns
        -------
            NTF dictionary:
                res["B"]: list[NDArray], NTF components in each dimension.

                res["n_iter"]: int, number of iterations performed.

                res["violation"]: float, violation. (fast hals can be reformulated \
                    as a projected gradient method).

                res["relative_error"]: float, model relative error.

            ISM dictionary:
                res['B']: list[NDArray]
                    B[0] contains ISM metascores.
                    B[1] contains list of view-mapping NDarrays.

                res['HHIi']: Number of non-negligable values in concatenated features by component.

                res['H_embed']: NDArray, NTF loadings in latent space.

                res['Q_embed']: NDArray, NTF view loadings.

                res['X_embed']: NDArray, Embedded views.

                res["relative_error"]: float, model relative error.        """
        verbose = self.verbose

        if isinstance(X, np.ndarray):
            # Validate X, B and update
            B, update = self._check_params_ntf(X, B, update)
            # Fast HALS
            if not np.isnan(X).any():
                if verbose >= 1:
                    print("Applying fast hals...")
                if B is None:
                    B = _initialize_ntf(
                        X,
                        self.n_components,
                        self.dim_order,
                        self.init,
                        self.use_randomized_svd,
                        random_state=self.random_state
                    )
                l1_reg, l2_reg = self._compute_regularization(X, alpha, l1_ratio)
                B, n_iter, violation, relative_error = _fit_coordinate_descent(
                    X,
                    B, 
                    update,
                    l1_reg, 
                    l2_reg,
                    self.tol,
                    self.max_iter,
                    self.random_state,
                    verbose,
                    self.target_polish,
                    self.target_polish_fraction,
                    self.target_polish_threshold,
                )
                res = {
                    "B": B,
                    "n_iter": n_iter,
                    "violation": violation,
                    "relative_error": relative_error
                }
                if verbose >= 1:
                    print("fast-hals applied.")
                return res
            else:
                warnings.warn('Applying random completions due to missing values in X.', UserWarning)
                if verbose >= 1:
                    print("Applying random completions with fast-hals...")
                B, relative_error = _fit_integrate_completions(
                    self,
                    X, 
                    B, 
                    update, 
                    alpha, 
                    l1_ratio,
                )
                res = {
                    "B": B,
                    "n_iter": np.nan,
                    "violation": np.nan,
                    "relative_error": relative_error
                }
                if verbose >= 1:
                    print(f"fast-hals applied with {self.n_completions} random completions.")
                return res            
        elif isinstance(X, list):
            # Validate X, B and update
            B, update = self._check_params_ism(X, B, update, self.H_mask)    
            
            if verbose >= 1:       
                print("Applying ism...")
            
            H_map, HHIi, B, embed, n_iter, error = _fit_integrate_sources(
                self,
                X,
                B,
                update,
            )
            res = {
                "B": [B, H_map],
                "HHIi": HHIi,
                "H_embed": embed.H,
                "Q_embed": embed.Q,
                "X_embed": embed.X,
                "n_iter": n_iter,
                "relative_error": error,
            }
            
            if verbose >= 1:
                print("ism applied.")
            return res
        else:
            raise ValueError(
                "Numpy ndarray or list of ndarrays must be passed to fit_transform. "
                f"An object of type {type(X)} was passed."
                )
        
    def _check_params_ntf(
        self,
        X: NDArray,
        B: Union[list[NDArray], None],
        update: Union[list[bool], None]
    ) -> NDArray:
        # Accept sparse formatted views

        try:
            X = check_array(
                X, 
                accept_sparse=("csr"), 
                dtype=[np.float64, np.float32],
                ensure_all_finite=self.force_all_finite,
                allow_nd=True,
            )
        except:
            X = check_array(
                X, 
                accept_sparse=("csr"), 
                dtype=[np.float64, np.float32],
                force_all_finite=self.force_all_finite,
                allow_nd=True,
            )
        check_non_negative(np.nan_to_num(X, nan=0.0), "X")
        if B is not None:
            if self.init != "custom":
                warnings.warn(
                    (
                        "When init!='custom', provided B list is ignored. Set "
                        " init='custom' to use it as initialization."
                    ),
                    RuntimeWarning,
                )
                B = None
            else:
                if isinstance(B, list):
                    if len(B) != X.ndim:
                        raise ValueError(
                            f"Initialialization list has length {len(B)}"
                            f" but the tensor has {X.ndim} dimensions."
                        )
                    avg = (np.nanmean(X) / self.n_components) ** (1 / X.ndim)
                    for dim in range(len(B)):
                        if B[dim] is None:
                            B[dim] = avg * np.ones((X.shape[dim], self.n_components))
                        _check_init(B[dim], (X.shape[dim], self.n_components), f"NTF (input B[{dim}])")
                else:
                    raise ValueError("B must be of type list[NDArrays].")        
        if update is not None:
            if len(update) != X.ndim:
                raise ValueError(
                    f"List passed to 'update' has wrong length. "
                    f"Expected {X.ndim}, but got {len(update)}."
                )
            for dim in range(len(update)):
                if not isinstance(update[dim], bool):
                    raise ValueError("update[{dim}] should be of type boolean.")
        else:
            update = [True for _ in range(X.ndim)]

        return B, update

    def _check_params_ism(
        self,
        X:list[NDArray],
        B: Union[NDArray, None],
        update: Union[bool, None],
        H_mask: Union[list[NDArray], None],
    ) -> NDArray:
        
        for view in range(len(X)):
            try:
                X[view] = check_array(
                    X[view], 
                    accept_sparse=False,
                    dtype=[np.float64, np.float32],
                    ensure_all_finite=False,
                    allow_nd=False,
                )
            except:
                X[view] = check_array(
                    X[view], 
                    accept_sparse=False,
                    dtype=[np.float64, np.float32],
                    force_all_finite=False,
                    allow_nd=False,
                )                
            # check all views share same dimension 0
            X_view_nan_to_0 = np.nan_to_num(X[view], nan=0.0)
            _check_init(
                        X_view_nan_to_0, 
                        (X[0].shape[0], X[view].shape[1]), 
                        f"ISM (input X[{view}])"
                    )
            check_non_negative(X_view_nan_to_0, f"X[{view}]")        
        
        if B is not None:
            if self.init != "custom":
                warnings.warn(
                    (
                        "When init!='custom', provided B list is ignored. Set "
                        " init='custom' to use it as initialization."
                    ),
                    RuntimeWarning,
                )
                B = None
            else:
                _check_init(B, (X.shape[0], self.n_components), f"ISM (input B)")

        if H_mask is not None:
            if self.n_embed != self.n_components:
                raise ValueError(
                    f"H_mask can't be applied with n_embed (={self.n_embed}) "
                    " != n_components (={self.n_components})."
                ) 
            if isinstance(H_mask, list):
                if len(H_mask) != len(X):
                    raise ValueError(
                        f"H mask has length {len(H_mask)}"
                        f" but there are {len(X)} views."
                    )
                for view in range(len(H_mask)):
                    _check_init(
                        H_mask[view], 
                        (X[view].shape[0], self.n_components), 
                        f"ISM (input H_mask[{view}])"
                    )
            else:
                raise ValueError("H_mask must be of type list[NDArrays].")        

        if update is not None:
            if not isinstance(update, bool):
                raise ValueError("update should be of type boolean.")
        else:
            update = True

        return B, update

    def _compute_regularization(
        self,
        X: NDArray, 
        alpha: Union[list[float], None] = None, 
        l1_ratio: float = 1.0
    ) -> Tuple[NDArray, NDArray]:
        """Compute scaled regularization terms
        The regularization terms are scaled by to keep their impact balanced
        with respect to one another.
        """
        n_dims = X.ndim

        if alpha is None:
            l1_reg = np.zeros(X.ndim)
        else:
            l1_reg = np.array(alpha)
            if l1_reg.shape[0] != n_dims:
                raise ValueError(
                    (
                        f"Regularization array has {l1_reg.shape[0]} elements"
                        f" but the tensor dimension is {n_dims}."
                    )
                )
            elif ((1 < l1_reg) | (l1_reg < 0)).any():
                raise ValueError("All regularization terms must be in range 0 to 1.")

        assert l1_ratio >= 0, "l1_ratio must be in range 0 to 1."
        assert l1_ratio <= 1, "l1_ratio must be in range 0 to 1."

        for dim in range(n_dims):
            shape_product = 1
            for j in range(n_dims):
                if j != dim:
                    shape_product *= X.shape[dim]
            l1_reg[dim] *= shape_product * l1_ratio
        
        l2_reg = l1_reg * (1.0 - l1_ratio)
        return (l1_reg, l2_reg)
