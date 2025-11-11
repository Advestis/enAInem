#  noqa: flake8
# Import env variables for the survival workflow
with open("config_survival.py") as f:
    config_code = f.read()

exec(config_code)

import warnings
import sys
import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection import GridSearchCV, KFold

# Load data and split in train / test
from sksurv.datasets import (  # type: ignore
    get_x_y,
    load_aids,
    load_arff_files_standardized,
    load_breast_cancer,
    load_flchain,
    load_gbsg2,
    load_veterans_lung_cancer,
    load_whas500,
)
from sksurv.functions import StepFunction
from sksurv.linear_model import CoxnetSurvivalAnalysis, CoxPHSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.nonparametric import kaplan_meier_estimator
from sksurv.preprocessing import OneHotEncoder, encode_categorical
from sksurv.util import Surv

from enainem import EnAInem

set_config(display="text")  # displays text representation of estimators
plt.rcParams["figure.figsize"] = [7.2, 4.8]
EPSILON = np.finfo(np.float64).eps
N_SEED_METHODS = len(NTF_SEED_NAMES) # type: ignore


# Utilities
def load_dsi(df, drop_columns=None):
    # Reset the index to move 'Time' and 'Exited' to columns
    df.reset_index(inplace=True)

    df.set_index("CustomerId", inplace=True)
    if drop_columns is not None:
        df.drop(columns=drop_columns, inplace=True)

    # Convert all object columns to category
    df[df.select_dtypes(include="object").columns] = df.select_dtypes(
        include="object"
    ).apply(lambda x: x.astype("category"))

    x, y = get_x_y(df, ["Exited", "Time"], pos_label=1, survival=True)

    return x, y

def load_ds1():
    df = pd.read_parquet(r".\data\ds1.parquet", engine="pyarrow", use_nullable_dtypes=False)
    return load_dsi(df, drop_columns=["RowNumber", "Surname"])


def load_ds2():
    df = pd.read_parquet(r".\data\ds2.parquet", engine="pyarrow", use_nullable_dtypes=False)
    df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'])
    df.drop('TotalCharges', axis=1, inplace=True)
    return load_dsi(df)


def load_ds3():
    df = pd.read_parquet(r".\data\ds3.parquet", engine="pyarrow", use_nullable_dtypes=False)
    return load_dsi(df)


def load_dataset(dataset_index):
    function_name = "load_" + DATASET_NAMES[dataset_index]  # type: ignore
    data_x, data_y = eval(f"{function_name}()")
    new_dtype = [("status", "?"), ("time", "<f8")]
    data_y = data_y.astype(new_dtype)
    # add small random number to time in case time is already discrete
    data_y["time"] += EPSILON * np.random.rand(data_y["time"].shape[0])

    # #add calibrator to normalize patterns
    # data_x['calibrator'] = np.random.rand(data_x.shape[0])
    # # Reorder columns to make 'calibrator' the first column
    # cols = ['calibrator'] + [col for col in data_x if col != 'calibrator']
    # data_x = data_x[cols]

    return data_x, data_y


def stepwise_regression(X_train, y_train, X_val, y_val, max_n_features=None):
    if max_n_features is None:
        max_n_features = X_train.shape[1]

    # Initialize variables
    selected_features = []
    remaining_features = list(range(X_train.shape[1]))
    best_score = 0
    improvement = True

    class EmptyCoxPHModel:
        pass

    # Stepwise selection
    while improvement and len(selected_features) < max_n_features:
        improvement = False
        scores_with_candidates = []
        # Try adding each remaining feature
        for feature in remaining_features:
            temp_features = selected_features + [feature]
            X_train_subset = X_train[:, temp_features]
            X_val_subset = X_val[:, temp_features]
            try:
                model = CoxPHSurvivalAnalysis().fit(X_train_subset, y_train)
                score = concordance_index_censored(
                    y_val["status"], y_val["time"], model.predict(X_val_subset)
                )[0]
            except Exception as e:
                model = EmptyCoxPHModel()
                score = 0

            scores_with_candidates.append((score, feature, model))

        # Select the best feature to add
        scores_with_candidates.sort(reverse=True)
        best_new_score, best_new_feature, best_new_model = scores_with_candidates[0]

        if best_new_score > best_score:
            best_score = best_new_score
            best_model = best_new_model
            selected_features.append(best_new_feature)
            remaining_features.remove(best_new_feature)
            if remaining_features:
                improvement = True

        # Check for features to remove
        for feature in selected_features:
            temp_features = [f for f in selected_features if f != feature]
            if temp_features:
                X_train_subset = X_train[:, temp_features]
                X_val_subset = X_val[:, temp_features]
                try:
                    model = CoxPHSurvivalAnalysis().fit(X_train_subset, y_train)
                    score = concordance_index_censored(
                        y_val["status"], y_val["time"], model.predict(X_val_subset)
                    )[0]
                except Exception as e:
                    model = EmptyCoxPHModel()
                    score = 0

                if score > best_score:
                    best_score = score
                    best_model = model
                    selected_features.remove(feature)
                    remaining_features.append(feature)
                    improvement = True

    return best_model, selected_features, best_score

def bootstrap_coxnet(best_model, X_train, y_train, selected_features, n_bootstrap=100):
    # Use max_iter=100 to speedup
    bootstrap_model = copy.deepcopy(best_model) 
    bootstrap_model.set_params(max_iter=1000)
    bootstrapped_coefs = []
    n_samples = len(y_train)

    # Fix the random seed
    np.random.seed(42)
    for _ in range(n_bootstrap):
        try:
            sample_indices = np.random.choice(np.arange(n_samples), size=n_samples, replace=True)
            X_sample = X_train[sample_indices]
            y_sample = y_train[sample_indices]
            bootstrap_model.fit(X_sample, y_sample)
            bootstrapped_coefs.append(bootstrap_model.coef_.ravel())
        except Exception as e:
            pass

    bootstrapped_coefs = np.array(bootstrapped_coefs)
    lower_conf_int = np.percentile(bootstrapped_coefs, 2.5, axis=0)[selected_features]
    upper_conf_int = np.percentile(bootstrapped_coefs, 97.5, axis=0)[selected_features]

    return lower_conf_int, upper_conf_int

def coxnet_regression(X_train, y_train):
    # determine the set of alphas which we want to evaluate
    l1_ratio = 0.9
    while l1_ratio >= 0:
        try:
            coxnet_model = CoxnetSurvivalAnalysis(n_alphas=10, l1_ratio=l1_ratio, max_iter=100)
            warnings.simplefilter("ignore", UserWarning)
            warnings.simplefilter("ignore", FitFailedWarning)
            coxnet_model.fit(X_train, y_train)
            estimated_alphas = coxnet_model.alphas_
            break
        except Exception as e:
            l1_ratio -= 0.1
    
    if l1_ratio < 0:
        print(f"\nCoxnet model did not converge.")
        sys.exit("Aborting the program...")

    # perform 5 fold cross-validation to estimate the performance – in terms of concordance index – 
    # for each alpha
    cv_random_state = 0
    max_cv_trials = 3
    while cv_random_state < max_cv_trials:
        try:
            cv = KFold(n_splits=5, shuffle=True, random_state=cv_random_state)
            gcv = GridSearchCV(
                CoxnetSurvivalAnalysis(l1_ratio=l1_ratio, fit_baseline_model=True, max_iter=1000),
                param_grid={"alphas": [[v] for v in estimated_alphas]},
                cv=cv,
                error_score=0.5,
                n_jobs=-1,
            ).fit(X_train, y_train)
            break
        except Exception as e:
            cv_random_state += 1

    if cv_random_state == max_cv_trials:
        # print(f"\n Max {max_cv_trials} trials in GridSearchCV reached.")
        # sys.exit("Aborting the program...")
        raise ValueError(f"\n Max {max_cv_trials} trials in GridSearchCV reached.")

    best_model = gcv.best_estimator_
    selected_features = np.nonzero(best_model.coef_)[0]
    selected_features_coef = best_model.coef_[selected_features].ravel()
    best_score = best_model.score(X_train, y_train)

    if BOOTSTRAP_COX: # type: ignore
        # Perform bootstrapping to get confidence intervals for the best model
        lower_conf_int, upper_conf_int = bootstrap_coxnet(best_model, X_train, y_train, selected_features)
        best_summary = list(zip(lower_conf_int, selected_features_coef, upper_conf_int))
    else:
        lower_conf_int, upper_conf_int = selected_features_coef, selected_features_coef
        best_summary = list(zip(lower_conf_int, selected_features_coef, upper_conf_int))

    return best_model, selected_features, best_score, best_summary

# Function to calculate normalization coefficients
def calculate_rss(df):
    non_missing_counts = df.count()
    df_filled = df.fillna(0)  # Fill NaNs with 0
    rss = np.sqrt(np.sum(df_filled**2, axis=0)) * (df.shape[0] / non_missing_counts)
    return rss


# Function to normalize using stored coefficients
def normalize_with_coefficients(df, coefficients):
    df_filled = df.fillna(0)  # Fill NaNs with 0
    return df_filled / coefficients


# Function to calculate the inverse probability of censoring weights
def ipcw(structured_data):
    # Estimate the Kaplan-Meier survival function for censoring times
    time, survival_prob = kaplan_meier_estimator(~structured_data["status"], structured_data["time"])  # type: ignore
    row_weights = np.sqrt(1 / np.interp(structured_data["time"], time, survival_prob))
    return row_weights


# Calculate mean sparsity of trained patterns
def h_sparsity(h):
    n_features = h.shape[0]
    n_components = h.shape[1]
    sparsity = 0

    for i in range(0, n_components):
        # calculate inverse hhi
        if np.max(h[:, i]) > 0:
            hhii = int(round(np.sum(h[:, i]) ** 2 / np.sum(h[:, i] ** 2)))
            sparsity += 1 - hhii / n_features

    sparsity /= n_components
    return sparsity


def model_error(X, A):
    return np.sum((X - _generate_tensor(A)) ** 2) / np.sum((X) ** 2)


# Return a tensor of the covariates where each observation is weighted
# by the probability of the event occurring within each layer/period
def create_weighted_tensor(coxph_model, data_x, w_nmf, percentiles, weights):
    if coxph_model is not None:
        # Predict the survival function for the specific patient
        survival_function = coxph_model.predict_survival_function(w_nmf)
        new_weights = np.zeros((data_x.shape[0], len(percentiles) + 1))

        for i in range(data_x.shape[0]):
            survival_at_t = np.array(
                [survival_function[i](percentiles[j]) for j in range(len(percentiles))]
            )
            survival_at_t_minus_1 = np.insert(survival_at_t, 0, 1)
            survival_at_t = np.append(survival_at_t, 0)
            # Calculate the hazard probability in each period
            new_weights[i, :] = np.sqrt(survival_at_t_minus_1 - survival_at_t)  # type: ignore

        if weights is not None:
            weights = np.sqrt(weights * new_weights)
        else:
            weights = new_weights

    else:
        weights = np.ones((data_x.shape[0], len(percentiles) + 1))
        temp = np.array(PERCENTILES_BOUNDS)  # type: ignore
        temp_1 = np.insert(temp, 0, 0)
        # temp = (np.append(temp, 100) - temp_1) / 100
        temp = np.sqrt((np.append(temp, 100) - temp_1) / 100)
        for j in range(len(percentiles) + 1):
            weights[:, j] = temp[j]

    # Create weighted tensor
    X = np.zeros((data_x.shape[0], data_x.shape[1], len(percentiles) + 1))
    for j in range(len(percentiles) + 1):
        X[:, :, j] = weights[:, j, np.newaxis] * data_x

    return X, weights


def plot_forest(summary, features, ax=None):
    if ax is None:
        ax = plt.gca()

    loghr = np.array(summary)[:,1]
    ci_lower = np.array(summary)[:,0]
    ci_upper = np.array(summary)[:,2]

    coef_order = np.argsort(np.abs(loghr))
    loghr = loghr[coef_order]
    ci_lower = ci_lower[coef_order]
    ci_upper = ci_upper[coef_order]
    ci_lower[ci_lower > loghr] = loghr[ci_lower > loghr]
    ci_upper[ci_upper < loghr] = loghr[ci_upper < loghr]

    ax.errorbar(
        loghr, range(len(loghr)), xerr=[loghr-ci_lower, ci_upper-loghr], fmt="o"
    )
    ax.axvline(x=0, linestyle="--", color="r")
    ax.set_yticks(range(len(loghr)))
    ax.set_yticklabels(features[coef_order])
    ax.set_xlabel("log(Hazard Ratio)")
    ax.set_title("Forest Plot for Coxnet Analysis")

    plt.tight_layout()
    plt.show()

def encode_split_norm(data_x, data_y, random_state, oneHot=False, impute=False):
    # Encode categorical columns and split into train, val and test sets

    # Encoding of all categorical columns
    if oneHot:
        data_x_numerical = OneHotEncoder().fit_transform(data_x)
    else:
        data_x_numerical = pd.get_dummies(data_x, dtype=float)

    if impute and data_x_numerical.isnull().any().any(): # type: ignore
        imputer = SimpleImputer(strategy='mean')
        imputed_array = imputer.fit_transform(data_x_numerical.to_numpy()) # type: ignore
        data_x_numerical = pd.DataFrame(imputed_array, columns=data_x_numerical.columns)   # type: ignore
    
    # First split: (train + validation) and test
    data_x_train_val, data_x_test, data_y_train_val, data_y_test = train_test_split(
        data_x_numerical,
        data_y,
        test_size=0.2,
        stratify=data_y["status"],
        random_state=random_state,
    )

    # Second split: train and validation
    data_x_train, data_x_val, data_y_train, data_y_val = train_test_split(
        data_x_train_val,
        data_y_train_val,
        test_size=0.2,
        stratify=data_y_train_val["status"],
        random_state=random_state,
    )
    
    # normalization of columns
    if oneHot:
        scaler = StandardScaler()
        scaler.fit(data_x_train)
        transformed_data = scaler.transform(data_x_train)
        data_x_train = pd.DataFrame(transformed_data, columns=data_x_train.columns) 
        transformed_data = scaler.transform(data_x_val)
        data_x_val = pd.DataFrame(transformed_data, columns=data_x_val.columns) 
        transformed_data = scaler.transform(data_x_test)
        data_x_test = pd.DataFrame(transformed_data, columns=data_x_test.columns) 
    else:
        normalization_coefficients = calculate_rss(data_x_train)
        data_x_train = normalize_with_coefficients(data_x_train, normalization_coefficients)
        data_x_val = normalize_with_coefficients(data_x_val, normalization_coefficients)
        data_x_test = normalize_with_coefficients(data_x_test, normalization_coefficients)

    return (
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
    )

def encode_categories_split_norm(data_x, data_y, random_state):
    (
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
    ) = encode_split_norm(data_x, data_y, random_state, oneHot=False, impute=SIMPLE_IMPUTER) # type: ignore

    max_n_components = min(data_x_train.shape[1], MAX_N_COMPONENTS)  # type: ignore

    return (
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
        max_n_components,
    )

def oneHotEncoder_split_norm(data_x, data_y, random_state):
    (
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
    ) = encode_split_norm(data_x, data_y, random_state, oneHot=True, impute=SIMPLE_IMPUTER) # type: ignore

    return (
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
    )

def setup_data_x_train_weighted_periods(data_x_train, data_y_train):
    # Setup train dataset for NTF with no censored observations
    # Apply the Inverse Probability of Censoring Weights
    data_x_train_no_cens = data_x_train.copy()
    for i in range(data_x_train_no_cens.shape[0]):
        if not data_y_train[i][0]:
            data_x_train_no_cens.iloc[i] = 0
    weights_ipcw = ipcw(data_y_train)
    data_x_train_no_cens = data_x_train_no_cens.multiply(weights_ipcw, axis=0)

    # Discretize survival_days
    survival_days = data_y_train["time"]
    percentiles = np.percentile(survival_days, PERCENTILES_BOUNDS)  # type: ignore
    survival_days_discretized = np.digitize(survival_days, percentiles)

    return data_x_train_no_cens, percentiles, survival_days_discretized


def nmf_based_model(
    data_x_train,
    data_x_val,
    data_x_test,
    data_y_train,
    data_y_val,
    data_y_test,
    max_n_components,
):
    # ------------------------------------------------------
    # NMF-based model
    # ------------------------------------------------------
    best_c_index_nmf = np.zeros(max_n_components)
    n_selected_components_nmf = np.zeros(max_n_components)
    best_model_nmf = [[]]
    selected_components_nmf = [[]]
    best_summary_nmf = [[]]
    w_train_nmf = [[]]
    w_val_nmf = [[]]
    w_test_nmf = [[]]
    h_nmf = [[]]

    for n_components in range(2, max_n_components + 1):
        nmtf = EnAInem(
            n_components=n_components,
            verbose=0,
            init="nndsvd",
            tol=1.0e-5,
            random_state=0,
            max_iter=200,
        )
        result = nmtf.fit_transform(data_x_train.values)
        w_train = result["B"][0]  # type: ignore

        h = result["B"][1]  # type: ignore
        h_nmf.append(h)  # type: ignore

        nmtf.init = "custom"

        result = nmtf.fit_transform(
            data_x_val.values,
            B=[np.ones((data_x_val.shape[0], n_components)), h],
            update=[True, False],
        )
        w_val = result["B"][0]  # type: ignore

        result = nmtf.fit_transform(
            data_x_test.values,
            B=[np.ones((data_x_test.shape[0], n_components)), h],
            update=[True, False],
        )
        w_test = result["B"][0]  # type: ignore

        scaler = StandardScaler()
        scaler.fit(w_train)
        w_train = scaler.transform(w_train)
        w_val = scaler.transform(w_val)
        w_test = scaler.transform(w_test)

        w_train_nmf.append(w_train)  # type: ignore
        w_val_nmf.append(w_val)  # type: ignore
        w_test_nmf.append(w_test)  # type: ignore

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            best_model, selected_components, best_c_index, best_summary = coxnet_regression(
                w_train, data_y_train
            )
            best_c_index_nmf[n_components - 1] = best_c_index
            n_selected_components_nmf[n_components - 1] = len(selected_components)
            best_model_nmf.append(best_model)  # type: ignore
            selected_components_nmf.append(selected_components) # type: ignore
            best_summary_nmf.append(best_summary)

    # Now take the best of nmf models and apply to the val & test data
    # print("")
    id_best_nmf = np.argmax(best_c_index_nmf)

    if USE_CONCORDANCE_INDEX_IPCW:  # type: ignore
        # Select observations in val & test with survival time compatible with train
        indices_test = np.where(data_y_test["time"] < data_y_train["time"].max())[0]
        best_c_index_nmf = concordance_index_ipcw(
            data_y_train,
            data_y_test[indices_test],
            best_model_nmf[id_best_nmf].predict( # type: ignore
                w_test_nmf[id_best_nmf][indices_test, :] # type: ignore
            ),
        )[0]  # type: ignore
    else:
        best_c_index_nmf = concordance_index_censored(
            data_y_test["status"],
            data_y_test["time"],
            best_model_nmf[id_best_nmf].predict( # type: ignore
                w_test_nmf[id_best_nmf]
            ),
        )[0]  # type: ignore

    return (
        best_c_index_nmf,
        n_selected_components_nmf[id_best_nmf],
        best_model_nmf[id_best_nmf],
        selected_components_nmf[id_best_nmf],
        best_summary_nmf[id_best_nmf],
        w_train_nmf[id_best_nmf],
        w_val_nmf[id_best_nmf],
        w_test_nmf[id_best_nmf],
        h_nmf[id_best_nmf],
    )


def cox_based_model(
    data_x_train_, data_x_val_, data_x_test_, data_y_train, data_y_val, data_y_test
):
    w_train_ = data_x_train_.values
    w_val_ = data_x_val_.values
    w_test_ = data_x_test_.values
    best_model_cox, selected_features_cox, _, best_summary_cox = coxnet_regression(
        w_train_, data_y_train
    )
    n_selected_features_cox = len(selected_features_cox)

    if USE_CONCORDANCE_INDEX_IPCW:  # type: ignore
        # Select observations in val & test with survival time compatible with train
        indices_test = np.where(data_y_test["time"] < data_y_train["time"].max())[0]
        best_c_index_cox = concordance_index_ipcw(
            data_y_train,
            data_y_test[indices_test],
            best_model_cox.predict(w_test_[indices_test, :]),
        )[0]  # type: ignore
    else:
        best_c_index_cox = concordance_index_censored(
            data_y_test["status"],
            data_y_test["time"],
            best_model_cox.predict(w_test_),  # type: ignore
        )[0]

    return (
        best_model_cox,
        n_selected_features_cox,
        selected_features_cox,
        best_c_index_cox,
        best_summary_cox,
    )


def ntf_loop_on_components(
    X_train_no_cens, X_train, X_val, X_test, data_y_train, data_y_val, max_n_components
):
    best_c_index_ntf = np.zeros(max_n_components)
    n_selected_components_ntf = np.zeros(max_n_components)
    best_model_ntf = [[]]
    selected_components_ntf = [[]]
    best_summary_ntf = [[]]
    w_train_ntf = [[]]
    w_val_ntf = [[]]
    w_test_ntf = [[]]
    h_ntf = [[]]
    q_ntf = [[]]

    for n_components in range(2, max_n_components + 1):
        nmtf = EnAInem(
            n_components=n_components,
            verbose=0,
            init="nndsvd",
            tol=1.0e-5,
            random_state=0,
            max_iter=200,
        )

        # Replace nan values by zeros (in effect cancelling the tensor row)
        X_train_no_cens = np.where(np.isnan(X_train_no_cens), 0, X_train_no_cens)

        result = nmtf.fit_transform(X_train_no_cens)
        h = result["B"][1]  # type: ignore
        q = result["B"][2]  # type: ignore

        # # TEST *******************************************************************
        # def generate_matrix(rows, n_components):
        #     matrix = np.random.rand(rows, n_components)
        #     matrix = matrix / np.sqrt(np.sum(matrix**2, axis=0))
        #     return matrix

        # rows = h.shape[0]
        # h = generate_matrix(rows, n_components)
        # # END TEST ***************************************************************

        h_ntf.append(h)  # type: ignore
        q_ntf.append(q)  # type: ignore

        nmtf.init = "custom"

        # print("X_train", np.isnan(X_train).any())
        result = nmtf.fit_transform(
            X_train,
            B=[None, h, q],
            update=[True, False, False],
        )
        w_train = result["B"][0]  # type: ignore

        # print("X_val", np.isnan(X_val).any())
        result = nmtf.fit_transform(
            X_val,
            B=[None, h, q],
            update=[True, False, False],
        )
        w_val = result["B"][0]  # type: ignore

        # print("X_test", np.isnan(X_test).any())
        result = nmtf.fit_transform(
            X_test,
            B=[None, h, q],
            update=[True, False, False],
        )
        w_test = result["B"][0]  # type: ignore

        scaler = StandardScaler()
        scaler.fit(w_train)
        w_train = scaler.transform(w_train)
        w_val = scaler.transform(w_val)
        w_test = scaler.transform(w_test)

        w_train_ntf.append(w_train)  # type: ignore
        w_val_ntf.append(w_val)  # type: ignore
        w_test_ntf.append(w_test)  # type: ignore

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            best_model, selected_components, best_c_index, best_summary = coxnet_regression(
                w_train, data_y_train
            )
            best_c_index_ntf[n_components - 1] = best_c_index
            n_selected_components_ntf[n_components - 1] = len(selected_components)
            best_model_ntf.append(best_model)  # type: ignore
            selected_components_ntf.append(selected_components) # type: ignore
            best_summary_ntf.append(best_summary)
            

    return (
        best_c_index_ntf,
        n_selected_components_ntf,
        best_model_ntf,
        selected_components_ntf,
        best_summary_ntf,
        w_train_ntf,
        w_val_ntf,
        w_test_ntf,
        h_ntf,
        q_ntf,
    )


def ntf_based_model(
    data_x_train_no_cens,
    best_model_nmf,
    w_train_nmf,
    w_val_nmf,
    w_test_nmf,
    selected_components_nmf,
    best_model_cox,
    data_x_train_,
    data_x_val_,
    data_x_test_,
    selected_features_cox,
    data_x_train,
    data_x_val,
    data_x_test,
    data_y_train,
    data_y_val,
    data_y_test,
    percentiles,
    survival_days_discretized,
    max_n_components,
):
    # Setup tensor of covariates using "supervized guess" to estimate ntf patterns
    X_train_no_cens = np.zeros((data_x_train.shape[0], data_x_train.shape[1], len(PERCENTILES_BOUNDS) + 1))  # type: ignore
    for i in range(X_train_no_cens.shape[0]):
        X_train_no_cens[i, :, survival_days_discretized[i]] = data_x_train_no_cens.iloc[
            i
        ]

    best_c_index_ntf_seed = np.zeros(N_SEED_METHODS)

    for n_iter_ntf in range(-N_SEED_METHODS, MAX_ITER_NTF + 1):  # type: ignore
        # skip tesing nmf and flat seed for the Lids conference
        if (n_iter_ntf != -N_SEED_METHODS + 1) and (n_iter_ntf < 0):
            continue
            # First 3 iterations evaluate ntf seeding methods
        if n_iter_ntf == -N_SEED_METHODS:
            # try nmf seed
            last_model = best_model_nmf
            last_w_train = w_train_nmf  # type: ignore
            last_w_val = w_val_nmf  # type: ignore
            last_w_test = w_test_nmf  # type: ignore
            weights_train = None
            weights_val = None
            weights_test = None
        elif n_iter_ntf == -N_SEED_METHODS + 1:
            # try cox seed
            last_model = best_model_cox
            last_w_train = data_x_train_.values
            last_w_val = data_x_val_.values
            last_w_test = data_x_test_.values
            weights_train = None
            weights_val = None
            weights_test = None
        elif n_iter_ntf == -N_SEED_METHODS + 2:
            # try flat seed
            last_model = None
            last_w_train = data_x_train.values
            last_w_val = data_x_val.values
            last_w_test = data_x_test.values
            weights_train = None
            weights_val = None
            weights_test = None
        elif n_iter_ntf == 0:
            # Select the best seed:
            # print(best_c_index_ntf_seed)
            best_seed = np.argmax(best_c_index_ntf_seed)
            if best_seed == 0:
                # use nmf seed
                last_model = best_model_nmf
                last_w_train = w_train_nmf  # type: ignore
                last_w_val = w_val_nmf  # type: ignore
                last_w_test = w_test_nmf  # type: ignore
                weights_train = None
                weights_val = None
                weights_test = None
            elif best_seed == 1:
                # use cox seed
                last_model = best_model_cox
                last_w_train = data_x_train_.values
                last_w_val = data_x_val_.values
                last_w_test = data_x_test_.values
                weights_train = None
                weights_val = None
                weights_test = None
            else:
                # use flat seed
                last_model = best_model_ntf[id_best_ntf]
                last_w_train = w_train_ntf[id_best_ntf]  # type: ignore
                last_w_val = w_val_ntf[id_best_ntf]  # type: ignore
                last_w_test = w_test_ntf[id_best_ntf]  # type: ignore
        else:
            last_model = best_model_ntf[id_best_ntf]
            last_w_train = w_train_ntf[id_best_ntf]  # type: ignore
            last_w_val = w_val_ntf[id_best_ntf]  # type: ignore
            last_w_test = w_test_ntf[id_best_ntf]  # type: ignore

        if n_iter_ntf > 0:
            # Save weights to check change between consecutive iterations
            weights_train_0 = weights_train.copy()  # type: ignore
            weights_val_0 = weights_val.copy()  # type: ignore
            weights_test_0 = weights_test.copy()  # type: ignore

        X_train, weights_train = create_weighted_tensor(
            last_model, data_x_train.values, last_w_train, percentiles, weights_train
        )
        X_val, weights_val = create_weighted_tensor(
            last_model, data_x_val.values, last_w_val, percentiles, weights_val
        )
        (X_test, weights_test,) = create_weighted_tensor(
            last_model, data_x_test.values, last_w_test, percentiles, weights_test
        )

        if n_iter_ntf > 0:
            weights_change = (
                np.sum((weights_train - weights_train_0) ** 2)
                + np.sum((weights_val - weights_val_0) ** 2)
                + np.sum((weights_test - weights_test_0) ** 2)
            ) / (
                np.sum(weights_train_0**2)
                + np.sum(weights_val_0**2)
                + np.sum(weights_test_0**2)
            )
            if weights_change < MIN_WEIGHTS_CHANGE:  # type: ignore
                break

        (
            best_c_index_ntf,
            n_selected_components_ntf,
            best_model_ntf,
            selected_components_ntf,
            best_summary_ntf,
            w_train_ntf,
            w_val_ntf,
            w_test_ntf,
            h_ntf,
            q_ntf,
        ) = ntf_loop_on_components(
            X_train_no_cens,
            X_train,
            X_val,
            X_test,
            data_y_train,
            data_y_val,
            max_n_components,
        )

        id_best_ntf = np.argmax(best_c_index_ntf)

        if USE_CONCORDANCE_INDEX_IPCW:  # type: ignore
            # Select observations in val & test with survival time compatible with train
            indices_val = np.where(data_y_val["time"] < data_y_train["time"].max())[0]
            indices_test = np.where(data_y_test["time"] < data_y_train["time"].max())[0]
            best_c_index_ntf_val = concordance_index_ipcw(
                data_y_train,
                data_y_val[indices_val],
                best_model_ntf[id_best_ntf].predict( # type: ignore
                    w_val_ntf[id_best_ntf][indices_val, :] # type: ignore
                ),
            )[0] 
        else:
            best_c_index_ntf_val = concordance_index_censored(
                data_y_val["status"],
                data_y_val["time"],
                best_model_ntf[id_best_ntf].predict( # type: ignore
                    w_val_ntf[id_best_ntf]
                ),
            )[0]  # type: ignore

        if n_iter_ntf < 0:
            best_c_index_ntf_seed[n_iter_ntf + N_SEED_METHODS] = best_c_index_ntf_val

    # Now take the best of ntf  models and apply to the test data

    id_best_ntf = np.argmax(best_c_index_ntf)

    if USE_CONCORDANCE_INDEX_IPCW:  # type: ignore
        best_c_index_ntf = concordance_index_ipcw(
            data_y_train,
            data_y_test[indices_test],
            best_model_ntf[id_best_ntf].predict( # type: ignore
                w_test_ntf[id_best_ntf][indices_test, :] # type: ignore
            ),
        )[0]  # type: ignore
        
    else:
        best_c_index_ntf = concordance_index_censored(
            data_y_test["status"],
            data_y_test["time"],
            best_model_ntf[id_best_ntf].predict( # type: ignore
                w_test_ntf[id_best_ntf]
            ),
        )[0]  # type: ignore

    return (
        best_c_index_ntf,
        n_selected_components_ntf[id_best_ntf],
        best_model_ntf[id_best_ntf],
        selected_components_ntf[id_best_ntf],
        w_train_ntf[id_best_ntf],
        w_val_ntf[id_best_ntf],
        w_test_ntf[id_best_ntf],
        h_ntf[id_best_ntf][:, selected_components_ntf[id_best_ntf]], # type: ignore
        q_ntf[id_best_ntf][:, selected_components_ntf[id_best_ntf]], # type: ignore
        best_summary_ntf[id_best_ntf],
        best_seed,
    )

def evaluate_models(data_x, data_y, random_state):
    # Evaluate different models

    # Description of the learning and testing approach:
    # 1. The dataset is first partitioned into train, validation, and test subsets.
    # 2. For a given number of components, train and validation datasets are used
    #    to retain only informative components with respect to time2event,
    #    using the c-index calculated on the validation data.
    # 3. At the end of the loop, the model with maximal c-index is chosen.
    # 4. Finally, the c-index of the chosen model is calculated on the test data.
    # Note that ntf/nmf rank are determined automatically.

    # Encode categorical columns, split and normalize
    (
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
        max_n_components,
    ) = encode_categories_split_norm(data_x, data_y, random_state)

    # Setup train dataset for NTF with no censored observations
    # Apply the Inverse Probability of Censoring Weights
    # Discretize survival_days
    (
        data_x_train_no_cens,
        percentiles,
        survival_days_discretized,
    ) = setup_data_x_train_weighted_periods(data_x_train, data_y_train)

    # print("\nnmf-based model")
    # ------------------------------------------------------
    # NMF-based model
    # ------------------------------------------------------
    (
        best_c_index_nmf,
        n_selected_components_nmf,
        best_model_nmf,
        selected_components_nmf,
        best_summary_nmf,
        w_train_nmf,
        w_val_nmf,
        w_test_nmf,
        h_nmf,
    ) = nmf_based_model(
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
        max_n_components,
    )

    # print("feature-based model")
    # ------------------------------------------------------
    # Feature-based model
    # ------------------------------------------------------

    # Encoding of all categorical columns skipping one category, split and normalize
    (
        data_x_train_,
        data_x_val_,
        data_x_test_,
        data_y_train,
        data_y_val,
        data_y_test,
    ) = oneHotEncoder_split_norm(data_x, data_y, random_state)

    (
        best_model_cox,
        n_selected_features_cox,
        selected_features_cox,
        best_c_index_cox,
        best_summary_cox,
    ) = cox_based_model(
        data_x_train_, data_x_val_, data_x_test_, data_y_train, data_y_val, data_y_test
    )

    # print("ntf-based model")
    # ------------------------------------------------------
    # NTF-based model
    # ------------------------------------------------------

    (
        best_c_index_ntf,
        n_selected_components_ntf,
        best_model_ntf,
        selected_components_ntf,
        w_train_ntf,
        w_val_ntf,
        w_test_ntf,
        h_ntf,
        q_ntf,
        best_summary_ntf,
        best_seed,
    ) = ntf_based_model(
        data_x_train_no_cens,
        best_model_nmf,
        w_train_nmf,
        w_val_nmf,
        w_test_nmf,
        selected_components_nmf,
        best_model_cox,
        data_x_train_,
        data_x_val_,
        data_x_test_,
        selected_features_cox,
        data_x_train,
        data_x_val,
        data_x_test,
        data_y_train,
        data_y_val,
        data_y_test,
        percentiles,
        survival_days_discretized,
        max_n_components,
    )

    return (
        n_selected_components_ntf,
        best_c_index_ntf,
        n_selected_components_nmf,
        best_c_index_nmf,
        n_selected_features_cox,
        data_x_train_.columns[selected_features_cox],
        best_c_index_cox,
        best_seed,
        data_x_train.columns,
        h_ntf,
        q_ntf,
        best_summary_cox,
        best_summary_ntf,
    )
