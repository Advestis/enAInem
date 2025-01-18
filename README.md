# EnAInem
**EnAInem** (pronounced /ɪˈneɪnəm/), is a Python class that implements state-of-the-art algorithms 
for decomposing nonnegative multiway array and multi-view data into rank-1 nonnegative tensors.
  
# Motivation
In machine learning, multi-view data refers to datasets comprising multiple distinct attribute sets 
("views") for the same set of observations. When these views share the same attributes but are observed 
in different contexts, the data can be represented as a tensor. **EnAInem** enables tensor decomposition 
using Non-Negative Tensor Factorization (NTF) and facilitates the integration of heterogeneous 
multi-view data into a shared latent space through the *Integrated Sources Model* (ISM).

# Installation
**EnAInem** is distributed as a standalone Python file, which can be downloaded and installed at your convenience.

# Dependencies
**EnAInem** has been tested with Python 3.11.9 and relies on the following libraries:
- **NumPy 2.0.2** (Numerical computing)
- **SciPy 1.13.1** (Scientific and technical computing)
- **Matplotlib 3.9.4** (Visualizations)
- **pandas 2.2.3** (Data manipulation and analysis)
- **scikit-learn 1.6.0** (Machine learning)
- **IPython 8.31.0** (Notebook interface)

All dependencies are listed in the `requirements.txt` file. To install them, run:

 `pip install -r requirements.txt` to have them all installed in your python environment.

While **EnAInem** may work with other versions of these libraries or in older Python environments, 
these configurations have not been tested. 

# Usage
The API strictly follows `scikit-learn` standards:
1. Create an instance of `EnAInem` class.
2. **EnAInem**'s `fit_transform` public method returns the decomposition in the form of a dictionary. The input type (multiway array or list of views) automatically routes `fit_transform` to either algorithm, NTF or ISM. See docstrings of the class and method for more details.
  ```python
  from enainem import EnAInem
  enainem = EnAInem(n_components=n_components)
  res = enainem.fit_transform(X)
  B = res["B"]
  relative_error = round(res["relative_error"], 4)
  print(f"relative_error = {relative_error}")
  ```
 
# Examples
Two simple examples (Python scripts) are provided:
- simple_ntf:
    - Generates a tensor `X_gen` from random basis vectors `B_gen`. 
    - Adds random noise to `X_gen` with an intensity defined by the `noise_coeff` parameter.
    - Estimates the basis vectors `B` using the `fit_transform` method from **EnAInem**.
    - When the `create_subplots` parameter is set to `True`, visualizations show the association 
      between the estimated base `B` and the original `B_gen`.
- simple_ism:
    - Generates a random non-negative matrix `X_1`.
    - Creates `X_2` by swapping the columns of `X_1` and adding noise.
    - Adds noise to `X_1`.
    - Applies `fit_transform` to the list of views `{X_1, X_2}` to recognize that both  
      convey the same information (up to noise), with the columns of `X_2` permuted.
    - When the `create_subplots` parameter is set to `True`, heatmaps visualize how the columns 
      of `X_1` and `X_2` load onto ISM components, highlighting the effective permutation.

#  References and Credits
If you use **EnAInem** for array or multi-view processing, please cite the following papers:
  - **Cichocki, A., & Phan, A.** (2009). 
  Fast Local Algorithms for Large Scale Nonnegative Matrix 
  and Tensor Factorizations. IEICE Trans. Fundam. Electron. 
  Commun. Comput. Sci., 92-A, 708-721.
  https://doi.org/10.1587/transfun.E92.A.708
- **Fogel, P., Geissler, C., Augé, F., Boldina, G., & Luta, G.** (2024). 
Integrated sources model: A new space-learning model for heterogeneous 
multi-view data reduction, visualization, and clustering. 
Artificial Intelligence in Health.
https://doi.org/10.36922/aih.3427

# License
This project is licensed under the MIT License.

# Authors
- **Paul Fogel** (paul.fogel@forvismazars.com)
- **George Luta**

# Acknowledgements
## **`scikit-learn`**
  - EnAInem extends the `BaseEstimator` class of `scikit-learn`'s `base` module, 
    leveraging its parameter validation mechanisms.
  - It also utilizes coordinate descent functions (`_update_coordinate_descent` and `_update_cdnmf_fast`), from the `_nmf` module implementing the fast HALS algorithm.
##  **ForvisMazars**
**EnAInem** is distributed through the R&D Forvis Mazars GitHub repository..
