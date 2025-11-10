# EnAInem

**EnAInem** (pronounced /ɪˈneɪnəm/) is a Python class that implements state-of-the-art algorithms for decomposing non-negative multiway array and multi-view data into rank-1 nonnegative tensors.

## Motivation

In machine learning, multi-view data refers to datasets comprising multiple distinct attribute sets ("views") for the same set of observations. When these views share the same attributes but are observed in different contexts, the data can be represented as a tensor. **EnAInem** enables tensor decomposition using Non-Negative Tensor Factorization (NTF) and allows the integration of heterogeneous multi-view data into a shared latent space through the *Integrated Sources Model* (ISM).

## Applications

**EnAInem** is mentioned in two recently published works:

- **Cox NTF**  
  [Cox NTF](https://www.advestis.com/post/coxntf-a-new-approach-for-joint-clustering-and-prediction-in-survival-analysis) is a breakthrough methodology that leverages Non-Negative Tensor Factorization to extract actionable latent factors directly linked to survival outcomes. It matches the predictive power of Coxnet models while offering a transparent, structured framework for clustering and segmentation. A tutorial (`survival_ntf_tutorial`) is provided in the root directory.

- **Target Polish**  
  [Target Polish](https://www.advestis.com/post/the-target-polish-a-new-approach-to-outlier-resistant-non-negative-matrix-and-tensor-factorization) is a robust and computationally efficient framework for Non-Negative Matrix and Tensor Factorization. It provides outlier resistance while maintaining the high performance of the fast-HALS algorithm used by EnAInem.

## Installation

**EnAInem** is distributed as a standalone Python package. To install it from source, run:

```bash
pip install .
```

To install optional development tools (e.g., for visualization and interactive use), run:

```bash
pip install .[dev]
```

## Dependencies

**EnAInem** has been tested with Python 3.11.9 and relies on the following core libraries:

- **NumPy 2.0.2** – Numerical computing
- **SciPy 1.13.1** – Scientific and technical computing
- **pandas 2.2.3** – Data manipulation and analysis
- **scikit-learn 1.6.0** – Machine learning
- **Dask 2024.8.0** – Parallel computing

Optional development tools (installable via `pip install .[dev]`) include:

- **Matplotlib 3.9.4** – Visualizations
- **IPython 8.31.0** – Notebook interface
- **pytest** – Testing framework

All dependencies are declared in `setup.py`. You can also install them manually using:

```bash
pip install -r requirements.txt
```

While **EnAInem** may work with other versions of these libraries or in older Python environments, these configurations have not been tested.

## Usage

The API strictly follows `scikit-learn` standards:

1. Create an instance of the `EnAInem` class.
2. Use the `fit_transform` method to perform decomposition. The input type (multiway array or list of views) automatically routes to either NTF or ISM.

```python
from enainem import EnAInem

enainem = EnAInem(n_components=n_components)
res = enainem.fit_transform(X)
B = res["B"]
relative_error = round(res["relative_error"], 4)
print(f"relative_error = {relative_error}")
```

## Examples

Two simple examples (Python scripts) are provided:

- **simple_ntf**:
  - Generates a tensor `X_gen` from random basis vectors `B_gen`.
  - Adds random noise to `X_gen` with intensity defined by `noise_coeff`.
  - Estimates the basis vectors `B` using `fit_transform`.
  - If `create_subplots=True`, visualizations show the association between estimated `B` and original `B_gen`.

- **simple_ism**:
  - Generates a random non-negative matrix `X_1`.
  - Creates `X_2` by permuting columns of `X_1` and adding noise.
  - Applies `fit_transform` to `[X_1, X_2]` to recognize shared structure.
  - If `create_subplots=True`, heatmaps visualize how columns of `X_1` and `X_2` load onto ISM components.

## References and Credits

If you use **EnAInem** for array or multi-view processing, please cite:

- **Cichocki, A., & Phan, A.** (2009).  
  Fast Local Algorithms for Large Scale Nonnegative Matrix and Tensor Factorizations.  
  *IEICE Trans. Fundam. Electron. Commun. Comput. Sci.*, 92-A, 708–721.  
  https://doi.org/10.1587/transfun.E92.A.708

- **Fogel, P., Geissler, C., Augé, F., Boldina, G., & Luta, G.** (2024).  
  Integrated sources model: A new space-learning model for heterogeneous multi-view data reduction, visualization, and clustering.  
  *Artificial Intelligence in Health*.  
  https://doi.org/10.36922/aih.3427

## License

This project is licensed under the MIT License.

## Authors

- **Paul Fogel** (paul.fogel@forvismazars.com)  
- **Christophe Geissler**  
- **George Luta**

## Acknowledgements

### `scikit-learn`
- EnAInem extends `BaseEstimator` from `scikit-learn`'s `base` module, leveraging its parameter validation.
- It uses coordinate descent functions (`_update_coordinate_descent`, `_update_cdnmf_fast`) from the `_nmf` module implementing the fast HALS algorithm.

### ForvisMazars
**EnAInem** is distributed through the R&D Forvis Mazars GitHub repository.