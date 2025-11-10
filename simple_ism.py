import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from enainem import EnAInem

# Install required packB_genes
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

def simple_ism(
    seed: int = 0,
    n_rows: int = 100,
    n_components: int = 10,
    max_noise_level:float = 0.1,
    create_subplots: bool = True,
):
    """
    Generate a random non-negative matrix `X_1`.
    - Swap the columns of `X_1` and add some noise to generate `X_2`.
    - Add some noise to `X_1`.
    - `fit_transform` is applied to the list of views `{X_1, X_2}` to recognize that `X_1` and `X_2` 
      convey the same information up to some noise, with the columns of `X_2` swapped around.
    - By setting the `create_subplots` parameter to `True`, heatmaps of the loadings of `X_1` and `X_2` 
      columns on ISM components show the effective permutation.

    seed: int = 0
        Seed used dto generate the tensor.
    
    n_rows: int = 100
        Number of rows in each view.
    
    n_components: int = 10
        Number of generative components.
    
    max_noise_level: float = 2.0
        Noise level to be added to the tensor.

    create_subplots: bool = False
        Visualize association between real and estimated generating factors.
    
    """
    import matplotlib.pyplot as plt

    np.random.seed(seed)
    # Generate a random non-negative matrix with 100 rows and 10 columns
    X_1 = np.random.rand(n_rows, n_components)
    # Swap the columns of X_1 and add some noise to generate X_2
    X_2 = np.random.permutation(X_1.T).T + np.random.uniform(low=0, high=max_noise_level, size=X_1.shape)
    # Add noise to X_1
    B = X_1.copy()
    X_1 += np.random.uniform(low=0, high=max_noise_level, size=X_1.shape)
    # ISM recognizes that X_1 and X_2 convey the same information up to some noise,
    # with the columns of X_2 swapped around. Heatmaps of the loadings of X_1 and X_2 columns
    # on ISM components show the effective permutation.

    enainem = EnAInem(
        n_components=n_components,
        init="nndsvd",
        tol=1e-6,
        verbose=False,
        n_embed=n_components,
        random_state=2,
    )

    X = [X_1, X_2]
    res = enainem.fit_transform(X)
    B = res['B']
    relative_error = round(res["relative_error"], 4)
    print(f"\nrelative_error={relative_error}")

    if create_subplots:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        ax[0].imshow(B[1][0], cmap='viridis', aspect='auto')
        # Add labels and title
        ax[0].set_xlabel('Component')
        ax[0].set_ylabel('Column')
        ax[0].set_title('Loadings of X_1 columns on ISM components')
        ax[1].imshow(B[1][1], cmap='viridis', aspect='auto')
        # Add labels and title
        ax[1].set_xlabel('Component')
        ax[1].set_ylabel('Column')
        ax[1].set_title('Loadings of X_2 columns on ISM components')

        # Show the plot
        plt.show()

    return(relative_error)

if __name__ == '__main__':
    simple_ism(create_subplots=True)
