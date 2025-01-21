import sys
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from enainem import EnAInem, _generate_tensor

# Install required packB_genes
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def simple_ntf(
    seed: int = 0,
    n_rows: list[int] = [50, 50, 50],
    n_components: int = 7,
    noise_coef:float = 0.5,
    create_subplots: bool = True,
):
    """
    Generate a tensor `X_gen` from random basis vectors `B_gen`. 
    - Random noise is added to `X_gen`. 
    Its intensity is defined by the `noise_coeff` parameter.
    - An estimate `B` of the basis vectors is obtained by the EnAInem `fit_transform` method.
    - By setting the `create_subplots` parameter to `True`, the association between
      the estimated base `B` and the original `B_gen` can be visualized.

    seed: int = 0
        Seed used to generate the tensor.
    
    n_rows: list[int] = [50, 50, 50]
        Shape of the tensor.
    
    n_components: int = 7
        Number of generative components.
    
    noise_coef: float = 2.0
        Multiplier of random generator of noise to be added to the tensor.

    create_subplots: bool = True
        Visualize association between real and estimated generating factors.
    
    """
    # Initialize random seed
    np.random.seed(seed=seed)

    # Notes:
    # n_completions==0|1 -> No random completion (fast HALS on imputed tensor) 
    # use_ism==False & nan_handler=="irc" & missing values -> NMF applied on concatenated W arrays
    # RANDOM_FOREST==True -> n_completions is automatically reset to 0 to apply fast HALS on imputed tensor
    # solver=="fast_hals" & nan_handler=="irc" & missing values -> random completions
    # solver=="fast_hals" & nan_handler=="mixed_hals" & missing values -> mixed hals
    # solver=="hals"|"mixed_hals" -> n_completions is automatically reset to 0 to apply weighted HALS

    # Instance EnAInem
    enainem = EnAInem(
        n_components=n_components,
        init="nndsvd",
        max_iter=200,
        tol=1e-6,
        verbose=0,
        norm_columns=0,
        random_state=0,
    )
    # enainem._validate_params()
    n_dims = len(n_rows)
    B_gen = [np.random.rand(n_rows[dim]*n_components).reshape(n_rows[dim], n_components) for dim in range(n_dims)]
    X_gen = _generate_tensor(B_gen)
    # Scaling to range [0, 1]
    X_gen = (X_gen - X_gen.min()) / (X_gen.max() - X_gen.min())
    X_gen += noise_coef*np.random.rand(np.prod(n_rows)).reshape(n_rows)
    X_gen.shape

    X = X_gen

    res = enainem.fit_transform(X)
    B = res["B"]
    relative_error = round(res["relative_error"], 4)
    print(f"relative_error = {relative_error}")

    if create_subplots:
        # Create subplots
        # ## Associate real and estimated components

        # Compute the correlation matrix between columns of B_gen[0] and B[0]
        correlation_matrix = np.cov(B_gen[0].T, B[0].T)[:B_gen[0].shape[1], B_gen[0].shape[1]:]

        # Find the optimal permutation using the Hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(-correlation_matrix)

        # Permute the columns of B[0] according to the optimal assignment
        B_permuted = [B[n][:, col_ind] for n in range(n_dims)]

        import matplotlib.pyplot as plt
        import warnings

        # Number of columns
        num_columns = B_gen[0].shape[1]

        warnings.filterwarnings("ignore", message="No artists with labels found to put in legend")
        fig, axes = plt.subplots(nrows=n_dims, ncols=num_columns, figsize=(15, 5))

        labels = [['W', 'W_hat'], ['H', 'H_hat'], ['Q', 'Q_hat']]
        for dim in range(n_dims):
            for i in range(num_columns):
                # Scatter plot
                axes[dim, i].scatter(B_gen[dim][:, i], B_permuted[dim][:, i])
                
                # Fit linear trend line
                coefficients = np.polyfit(B_gen[dim][:, i], B_permuted[dim][:, i], 1)
                linear_fit = np.poly1d(coefficients)
                
                # Generate trend line values
                trend_x = np.linspace(min(B_gen[dim][:, i]), max(B_gen[dim][:, i]), 100)
                trend_y = linear_fit(trend_x)
                
                # Plot trend line
                axes[dim, i].plot(trend_x, trend_y, color='red', label='_Trend line')

                # Add title and labels
                axes[dim, i].set_title(f'Component {i+1}')
                axes[dim, i].set_xlabel(labels[dim][0])
                axes[dim, i].set_ylabel(labels[dim][1])
                axes[dim, i].legend()

        plt.tight_layout()
        plt.show()

    return(relative_error)

if __name__ == '__main__':
    simple_ntf(create_subplots=True)
