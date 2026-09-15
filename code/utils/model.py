import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def remove_dependent_columns(X, tolerance=None):

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from scipy.linalg import qr

    X = X.copy()

    # Ensure numeric values
    X = X.apply(pd.to_numeric, errors="coerce")

    # Replace invalid values
    X = X.replace([np.inf, -np.inf], np.nan)

    # Remove rows containing invalid values
    X = X.dropna()

    # Remove constant columns
    constant_cols = X.columns[X.nunique() <= 1].tolist()
    X = X.drop(columns=constant_cols)

    if X.shape[1] == 0:
        return X, constant_cols, []

    matrix = X.to_numpy(dtype=float)

    # QR decomposition with column pivoting
    Q, R, pivot_order = qr(matrix, mode="economic", pivoting=True)

    # Determine numerical rank
    if tolerance is None:
        tolerance = np.finfo(float).eps * max(matrix.shape) * abs(R[0, 0])

    rank = np.sum(np.abs(np.diag(R)) > tolerance)

    # Columns after the rank are linearly dependent
    dependent_positions = pivot_order[rank:]
    dependent_cols = X.columns[dependent_positions].tolist()

    X_fixed = X.drop(columns=dependent_cols)

    return X_fixed, constant_cols, dependent_cols

def logistic_woe_run(df_tmp,group_i):

    from utils import model as mod
    import statsmodels.api as sm
    from utils import model as mod

    # check separtion and remove those columns
    flagged_variables = mod.find_separation_variables(
        df=df_tmp,
        target="denied",
        print_results=False,
        check_near_separation=True
    )

    df_tmp.drop(columns = flagged_variables, inplace = True)


    #############create woe
    import toad

    # Ensure object columns are clean strings
    object_cols = df_tmp.select_dtypes(include=["object", "string"]).columns

    for col in object_cols:
        df_tmp[col] = df_tmp[col].astype("string").fillna("__MISSING__")



    # Create the combiner
    combiner = toad.transform.Combiner()

    # Fit bins 
    combiner.fit(
        df_tmp,
        y='denied',
        method="chi",
        min_samples=0.05,
        empty_separate=True, #edit
        exclude= [group_i]   #['denied']
    )

    # Apply bin rules
    df_tmp_bin = combiner.transform(df_tmp)

    # convert bins to woe 
    woe_transformer = toad.transform.WOETransformer()

    df_tmp_woe = woe_transformer.fit_transform(
        df_tmp_bin,
        df_tmp_bin['denied'],
        exclude=['denied',group_i]
    )


    ####### assign x and y data
    y = df_tmp_woe["denied"]

    # drop columns with single value. will end up being things like indicator for segment
    df_tmp_woe = df_tmp_woe.loc[:, df_tmp_woe.nunique(dropna=False) > 1]

    # reorder columns
    col_order = df_tmp_woe.columns.tolist()
    col_order.remove(group_i)
    df_tmp_woe = df_tmp_woe[[group_i] + col_order]

    # X = sm.add_constant(df_tmp_woe.drop(columns='denied'))

    X = df_tmp_woe.drop(columns='denied')

    # test and fix rank

    X_fixed, constant_cols, dependent_cols = mod.remove_dependent_columns(X)

    print("Removed constant columns:", constant_cols)
    print("Removed dependent columns:", dependent_cols)

    # Align y with rows remaining after cleaning X
    y_fixed = y.loc[X_fixed.index]



    model = sm.Logit(y_fixed, sm.add_constant(X_fixed))
    result = model.fit()

    print(result.summary())

    return result

def ols_dummy_run(df_tmp,group_i):
    import numpy as np
    import pandas as pd
    import statsmodels.api as sm


    # df_tmp.info()

    # drop denied
    df_tmp.drop(columns=['action_taken'], inplace= True)

    # check unique values in characaters. we cant have too many and end up losing dfs

    def remove_high_cardinality_text_columns(df, max_cnt):
        """
        Remove object and string columns with more than max_cnt
        unique values.

        Returns:
            cleaned_df, unique_counts, removed_columns
        """
        text_cols = df.select_dtypes(
            include=["object", "str"]
        ).columns

        unique_counts = df[text_cols].nunique()

        removed_columns = unique_counts[
            unique_counts > max_cnt
        ].index.tolist()

        cleaned_df = df.drop(columns=removed_columns)

        return cleaned_df, unique_counts, removed_columns


    df_tmp, counts, removed = remove_high_cardinality_text_columns(
        df_tmp,
        max_cnt=20
    )

    # print(counts)
    print("Removed columns:", removed)



    # create bins for numerics


    import pandas as pd

    def qcut_numeric_columns(df, n_bins=5, exclude_cols=None):
        df = df.copy()

        if exclude_cols is None:
            exclude_cols = []

        numeric_cols = [
            col for col in df.select_dtypes(include="number").columns
            if col not in exclude_cols
        ]

        for col in numeric_cols:
            df[col] = (
                pd.qcut(
                    df[col],
                    q=n_bins,
                    labels=False,
                    duplicates="drop"
                )
                .add(1)
                .astype("object")
                .fillna("missing")
            )

        return df


    df_tmp = qcut_numeric_columns(
        df_tmp,
        n_bins=3,
        exclude_cols=["interest_rate", group_i]
    )



    # df_tmp.head()

    # df_tmp.info()



    # make dummy vars

    df_tmp = pd.get_dummies(
        df_tmp,
        columns=df_tmp.select_dtypes(exclude="number").columns,
        drop_first=True,
        dtype=int
    )

    df_tmp.shape

    ####### assign x and y data
    y = df_tmp["interest_rate"]

    # drop columns with single value. will end up being things like indicator for segment
    df_tmp = df_tmp.loc[:, df_tmp.nunique(dropna=False) > 1]





    # reorder columns
    col_order = df_tmp.columns.tolist()
    col_order.remove(group_i)
    df_tmp = df_tmp[[group_i] + col_order]

    X = sm.add_constant(df_tmp.drop(columns='interest_rate'))


    # Duplicate columns
    duplicate_cols = X.columns[X.T.duplicated()].tolist()
    print("Duplicate columns:", duplicate_cols)

    # Constant columns
    constant_cols = X.columns[X.nunique() <= 1].tolist()
    print("Constant columns:", constant_cols)

    # Matrix rank
    print("Number of columns:", X.shape[1])
    print("Matrix rank:", np.linalg.matrix_rank(X.to_numpy()))

    # do if duplicates then drop them

    if len(duplicate_cols) > 0:
        X = X.drop(columns=constant_cols + duplicate_cols)


    # drop high corr

    import numpy as np

    corr = X.corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > 0.95)
    ]

    X = X.drop(columns=to_drop)

    print("Dropped columns:", to_drop)






    model = sm.OLS(y, X)
    result = model.fit()

    # print(result.summary())

    return result

def find_separation_variables(
    df,
    target,
    print_results=True,
    check_near_separation=True,
    n_numeric_bins=10
):
    """
    Find variables that may cause complete or near separation
    in a binary logistic-regression model.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data.

    target : str
        Binary target column containing exactly two classes.

    print_results : bool, default=True
        Whether to print details about flagged variables.

    check_near_separation : bool, default=True
        For numeric variables, also flag bins with a target rate
        of exactly 0 or 1.

    n_numeric_bins : int, default=10
        Number of quantile bins used when checking near separation.

    Returns
    -------
    flagged_variables : list
        List of variable names that may have separation.
    """

    import numpy as np
    import pandas as pd


    if target not in df.columns:
        raise ValueError(f"Target column '{target}' was not found.")

    target_values = df[target].dropna().unique()

    if len(target_values) != 2:
        raise ValueError(
            f"'{target}' must contain exactly two classes. "
            f"Found: {target_values}"
        )

    flagged_variables = []
    details = {}

    # Identify variable types
    categorical_cols = df.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()

    numeric_cols = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_cols = [
        col for col in categorical_cols
        if col != target
    ]

    numeric_cols = [
        col for col in numeric_cols
        if col != target
    ]

    # ---------------------------------------------------------
    # Check categorical variables
    # ---------------------------------------------------------
    for col in categorical_cols:
        temp = df[[col, target]].copy()

        # Treat missing values as their own category
        temp[col] = temp[col].astype("string").fillna("__MISSING__")

        counts = pd.crosstab(temp[col], temp[target])

        # Make sure both target classes are represented as columns
        for target_class in target_values:
            if target_class not in counts.columns:
                counts[target_class] = 0

        counts = counts[list(target_values)]

        # A level containing only one target class is separated
        separated_levels = counts.index[
            counts.min(axis=1) == 0
        ].tolist()

        if separated_levels:
            if col not in flagged_variables:
                flagged_variables.append(col)

            details[col] = {
                "type": "categorical",
                "reason": "One or more levels contain only one target class",
                "levels": separated_levels,
                "counts": counts.loc[separated_levels]
            }

    # ---------------------------------------------------------
    # Check numeric variables
    # ---------------------------------------------------------
    for col in numeric_cols:
        temp = df[[col, target]].dropna().sort_values(col)

        x = temp[col].to_numpy()
        y = temp[target].to_numpy()

        # Need at least two distinct values
        if len(np.unique(x)) < 2:
            continue

        # Check every possible cutoff between distinct values
        candidate_positions = np.where(np.diff(x) != 0)[0]
        exact_cutoffs = []

        for position in candidate_positions:
            cutoff = (x[position] + x[position + 1]) / 2

            lower_values = y[x <= cutoff]
            upper_values = y[x > cutoff]

            # Perfect separation if each side contains one class only
            if (
                len(np.unique(lower_values)) == 1
                and len(np.unique(upper_values)) == 1
                and lower_values[0] != upper_values[0]
            ):
                exact_cutoffs.append(cutoff)

        near_separation_bins = None

        # Optional check for bins with event rate exactly 0 or 1
        if check_near_separation:
            try:
                temp = temp.copy()
                temp["bin"] = pd.qcut(
                    temp[col],
                    q=n_numeric_bins,
                    duplicates="drop"
                )

                grouped = temp.groupby(
                    "bin",
                    observed=False
                )[target].agg(
                    count="size",
                    target_rate="mean"
                )

                near_separation_bins = grouped[
                    (grouped["target_rate"] == 0) |
                    (grouped["target_rate"] == 1)
                ]

            except ValueError:
                near_separation_bins = None

        has_near_separation = (
            near_separation_bins is not None
            and not near_separation_bins.empty
        )

        if exact_cutoffs or has_near_separation:
            flagged_variables.append(col)

            details[col] = {
                "type": "numeric",
                "exact_separation_cutoffs": exact_cutoffs,
                "near_separation_bins": near_separation_bins
            }

    # ---------------------------------------------------------
    # Optional printing
    # ---------------------------------------------------------
    if print_results:
        if not flagged_variables:
            print("No separation variables were found.")
        else:
            print("Variables with possible separation:")

            for col in flagged_variables:
                print(f"\n{col}")

                variable_details = details[col]

                if variable_details["type"] == "categorical":
                    print(
                        "  Reason:",
                        variable_details["reason"]
                    )
                    print(
                        "  Levels:",
                        variable_details["levels"]
                    )
                    print(variable_details["counts"])

                elif variable_details["type"] == "numeric":
                    cutoffs = variable_details[
                        "exact_separation_cutoffs"
                    ]

                    if cutoffs:
                        print(
                            "  Exact separation cutoff(s):",
                            cutoffs
                        )

                    bins = variable_details[
                        "near_separation_bins"
                    ]

                    if bins is not None and not bins.empty:
                        print("  Bins with target rate 0 or 1:")
                        print(bins)

    return flagged_variables

def make_match_pair(df_tmp, target_to_exclude,group_i):

    from psmpy import PsmPy
    from psmpy.functions import cohenD
    # from psmpy.plotting import *

    # redo this but wit the pre woe data. need to redo woe before matched pair reg
    # df_tmp
    # df_tmp_woe


    # df_tmp_woe["row_id"] = df_tmp_woe.index
    df_tmp["row_id"] = df_tmp.index

    # df_tmp.head()

    # need to make dummies
    df_tmp_4psa = df_tmp



    exclude = [target_to_exclude, group_i, 'row_id']

    df_tmp_4psa = pd.concat(
        [
            df_tmp_4psa[exclude],
            pd.get_dummies(df_tmp_4psa.drop(columns=exclude), dtype=int)
        ],
        axis=1
    )




    # df_tmp_4psa.info()

    missing_columns = df_tmp_4psa.columns[df_tmp_4psa.isna().any()].tolist()
    missing_columns

    # target_to_exclude + missing_columns



    # psm = PsmPy(df_tmp_woe, treatment=group_i, indx='row_id', exclude = ['denied'])
    psm = PsmPy(df_tmp_4psa, treatment=group_i, indx='row_id', exclude = [target_to_exclude] + missing_columns)

    psm.logistic_ps(balance = True)

    # psm.kdtree_matched(matcher='propensity_logit', replacement=False, caliper=None, drop_unmatched=True)
    # psm.kdtree_matched_12n(matcher='propensity_logit', how_many=1)
    psm.kdtree_matched(
        matcher="propensity_score",
        replacement=True,
        caliper=0.2,
        drop_unmatched=True
    )


    # psm.df_matched

    # get unbinned attributes 
    # df_tmp_psa = pd.concat([], axis=1)
    df_tmp_psa = pd.DataFrame(psm.df_matched['row_id']).merge(df_tmp, on="row_id", how="left")
    df_tmp_psa.drop(columns='row_id', inplace = True)

    # df_tmp_psa.head()

    return df_tmp_psa

def corr_crit(corr_tmp, thresh):

    import pandas as pd

    corr_ind = pd.DataFrame(columns = ['var','corr','corr ind'])

    # thresh = .3

    for i in corr_tmp.columns:
        # print(i)

        # mx = corr_tmp[i].max()
        mx = corr_tmp[i].drop(index=i).max()
        mi = corr_tmp[i].drop(index=i).min()

        if abs(mx) > abs(mi):
            res = mx
        else:
            res = mi

        # print(res)

        if abs(res) > thresh:
            indy = True
        else:
            indy = False

        new_row = pd.DataFrame([{
        "var": i,
        'corr': res,
        "corr ind": indy
        }])

        corr_ind = pd.concat([corr_ind, new_row], ignore_index=True)

    # break

    return corr_ind

def calculate_auc_statistics(
    df,
    column_names,
    event_column
):
    """
    Calculate one univariate AUC per input variable.

    Numeric variables are tested directly.
    Categorical variables are one-hot encoded internally.
    """

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    required_columns = column_names + [event_column]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The following columns are missing: {missing_columns}"
        )

    auc_results = []

    for column in column_names:

        model_df = df[[column, event_column]].dropna()

        # Convert categorical columns to dummy variables
        X = pd.get_dummies(
            model_df[[column]],
            drop_first=True,
            dtype=float
        )

        y = model_df[event_column].astype(float)

        # Skip variables with no variation
        if X.shape[1] == 0:
            continue

        X = sm.add_constant(X)

        try:
            model = sm.Logit(y, X).fit(disp=False)

            probabilities = model.predict(X)

            auc = roc_auc_score(
                y,
                probabilities
            )

            auc_results.append({
                "var": column,
                "auc": auc
            })

        except Exception as error:
            print(f"Could not fit model for '{column}': {error}")

    aucs = (
        pd.DataFrame(auc_results)
        .sort_values(by="auc", ascending=False)
        .reset_index(drop=True)
    )

    return aucs

def calculate_wald_statistics(
    df,
    column_names,
    event_column
):
    """
    Calculate one Wald statistic per input variable.

    Numeric variables are tested directly.
    Categorical variables are one-hot encoded internally and tested jointly.
    """

    import numpy as np
    import pandas as pd
    import statsmodels.api as sm

    required_columns = column_names + [event_column]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The following columns are missing: {missing_columns}"
        )

    wald_results = []

    for column in column_names:

        model_df = df[[column, event_column]].dropna()

        # Convert categorical columns to dummy variables
        X = pd.get_dummies(
            model_df[[column]],
            drop_first=True,
            dtype=float
        )

        y = model_df[event_column].astype(float)

        # Skip variables with no variation
        if X.shape[1] == 0:
            continue

        X = sm.add_constant(X)

        try:
            model = sm.Logit(y, X).fit(disp=False)

            predictor_columns = list(X.columns)
            predictor_columns.remove("const")

            # Position of predictor columns in model.params
            predictor_indices = [
                predictor_columns.index(col) + 1
                for col in predictor_columns
            ]

            # Constraint matrix for a joint Wald test
            constraint_matrix = np.zeros(
                (len(predictor_indices), len(model.params))
            )

            for row, index in enumerate(predictor_indices):
                constraint_matrix[row, index] = 1

            # Joint Wald statistic for the variable
            wald_test = model.wald_test(
                constraint_matrix,
                scalar=True
            )

            wald_statistic = float(wald_test.statistic)

            wald_results.append({
                "var": column,
                "wald": wald_statistic
            })

        except Exception as error:
            print(f"Could not fit model for '{column}': {error}")

    walds = (
        pd.DataFrame(wald_results)
        .sort_values(by="wald", ascending=False)
        .reset_index(drop=True)
    )

    return walds

def plot_event_num(
    df,
    column_names,
    event_column,
    n_bins=5
):
    """
    Plot histograms and event rates for multiple numerical columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input dataframe.

    column_names : list
        List of numerical column names to plot.

    event_column : str
        Column containing the event indicator, where 1 represents an event.

    n_bins : int
        Number of bins for each variable.
    """

    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt



    n_cols = 3

    if not isinstance(column_names, list):
        raise TypeError("column_names must be a list of column names.")

    required_columns = column_names + [event_column]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The following columns are missing from the dataframe: "
            f"{missing_columns}"
        )

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1.")

    n_plots = len(column_names)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax1, column in zip(axes, column_names):

        # Remove missing values from the variable and event column
        plot_df = df[[column, event_column]].dropna().copy()

        # Create equally spaced bins
        bin_edges = np.linspace(
            plot_df[column].min(),
            plot_df[column].max(),
            n_bins + 1
        )

        # Handle columns with only one unique value
        if len(np.unique(bin_edges)) < 2:
            ax1.hist(
                plot_df[column],
                bins=1,
                color="skyblue",
                edgecolor="black"
            )
            ax1.set_title(column)
            ax1.set_xlabel(column)
            ax1.set_ylabel("Number of observations")
            continue

        # Assign observations to bins
        plot_df["bin"] = pd.cut(
            plot_df[column],
            bins=bin_edges,
            include_lowest=True
        )

        # Calculate event rate in each bin
        event_rates = (
            plot_df
            .groupby("bin", observed=False)[event_column]
            .mean()
        )

        # Calculate bin centers
        bin_centers = np.array([
            interval.mid for interval in event_rates.index
        ])

        # Histogram
        ax1.hist(
            plot_df[column],
            bins=bin_edges,
            color="skyblue",
            edgecolor="black",
            alpha=0.7
        )

        ax1.set_xlabel(column)
        ax1.set_ylabel("Number of observations")
        ax1.set_title(f"{column}: distribution and event rate")

        # Event-rate line on a secondary y-axis
        ax2 = ax1.twinx()

        ax2.plot(
            bin_centers,
            event_rates.values,
            color="red",
            marker="o",
            linewidth=2,
            label="Event rate"
        )

        ax2.set_ylabel("Event rate")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )

    # Hide unused subplot axes
    for ax in axes[n_plots:]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()

    return fig, axes

def plot_event_cat(
    df,
    column_names,
    event_column="bad"
):


    import math
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    n_cols = 3
    n_plots = len(column_names)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7 * n_cols, 5 * n_rows),
        squeeze=False
    )

    axes = axes.flatten()

    for ax1, column in zip(axes, column_names):

        plot_df = df[[column, event_column]].copy()

        # Treat missing values as their own category
        plot_df[column] = plot_df[column].fillna("Missing")

        summary = (
            plot_df
            .groupby(column, observed=False)[event_column]
            .agg(
                count="count",
                event_rate="mean"
            )
            .reset_index()
        )

        # Bar chart: number of observations
        ax1.bar(
            summary[column].astype(str),
            summary["count"],
            color="skyblue",
            edgecolor="black",
            alpha=0.7
        )

        ax1.set_xlabel(column)
        ax1.set_ylabel("Number of observations")
        ax1.set_title(f"{column}: distribution and event rate")
        ax1.tick_params(axis="x", rotation=45)

        # Line chart: event rate
        ax2 = ax1.twinx()

        ax2.plot(
            summary[column].astype(str),
            summary["event_rate"],
            color="red",
            marker="o",
            linewidth=2
        )

        ax2.set_ylabel("Event rate")
        ax2.set_ylim(0, 1)
        ax2.yaxis.set_major_formatter(
            plt.FuncFormatter(lambda value, _: f"{value:.0%}")
        )

    # Hide unused plots in the last row
    for ax in axes[len(column_names):]:
        ax.set_visible(False)

    fig.tight_layout()
    plt.show()

    return fig, axes

def cv_auc_logit(X, y, features, cv):
    """Return mean cross-validated AUC for a statsmodels logistic model."""

    import numpy as np
    import statsmodels.api as sm
    from sklearn.metrics import roc_auc_score

    aucs = []

    for train_idx, test_idx in cv.split(X, y):
        X_train = sm.add_constant(
            X.iloc[train_idx][features],
            has_constant="add"
        )
        X_test = sm.add_constant(
            X.iloc[test_idx][features],
            has_constant="add"
        )

        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        try:
            model = sm.Logit(y_train, X_train).fit(
                disp=False,
                maxiter=200
            )

            predictions = model.predict(X_test)
            aucs.append(roc_auc_score(y_test, predictions))

        except Exception:
            # Handles singular matrices or failed model fits
            aucs.append(np.nan)

    return np.nanmean(aucs)

def forward_select_auc(X, y, min_improvement=0.001, random_state=42):

    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state
    )

    remaining = list(X.columns)
    selected = []

    best_auc = 0.5
    history = []

    while remaining:
        results = []

        for candidate in remaining:
            features = selected + [candidate]

            auc = cv_auc_logit(
                X=X,
                y=y,
                features=features,
                cv=cv
            )

            results.append((auc, candidate))

        candidate_auc, best_candidate = max(results)

        if candidate_auc > best_auc + min_improvement:
            selected.append(best_candidate)
            remaining.remove(best_candidate)
            best_auc = candidate_auc

            history.append({
                "added": best_candidate,
                "features": selected.copy(),
                "cv_auc": best_auc
            })

            print(
                f"Added {best_candidate:30s} "
                f"CV AUC = {best_auc:.4f}"
            )
        else:
            break

    return selected, pd.DataFrame(history)

def plot_logit_marginal_effect(
    model,
    data,
    variable,
    selected_features,
    grid_size=100,
    ax=None
):
    """
    Plot the marginal effect of `variable`, holding all other selected
    features at their sample means.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if variable not in selected_features:
        raise ValueError(
            f"{variable!r} must be included in selected_features."
        )

    missing = [
        feature for feature in selected_features
        if feature not in data.columns
    ]

    if missing:
        raise ValueError(
            f"These selected features are not in data: {missing}"
        )

    # All selected features except the variable being plotted
    variables_to_hold = [
        feature for feature in selected_features
        if feature != variable
    ]

    # Values over which to evaluate the target variable
    x_values = np.linspace(
        data[variable].min(),
        data[variable].max(),
        grid_size
    )

    # Create prediction data
    new_data = pd.DataFrame({
        variable: x_values
    })

    # Hold all remaining selected features at their means
    for feature in variables_to_hold:
        new_data[feature] = data[feature].mean()

    # Add any model terms not explicitly listed, if needed
    exog_names = model.model.exog_names

    for name in exog_names:
        if name != "const" and name not in new_data.columns:
            if name in data.columns:
                new_data[name] = data[name].mean()

    # Match the exact order used when fitting the model
    exog = new_data.reindex(
        columns=exog_names,
        fill_value=1
    )

    # Predicted probabilities
    probabilities = model.predict(exog)

    # Logistic marginal effect:
    # dP/dx = p(1-p) * beta
    beta = model.params[variable]

    marginal_effects = (
        probabilities * (1 - probabilities) * beta
    )

    results = pd.DataFrame({
        variable: x_values,
        "predicted_probability": probabilities,
        "marginal_effect": marginal_effects
    })

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10))

    ax.plot(
        results[variable],
        results["marginal_effect"],
        linewidth=2
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.set_xlabel(variable)
    ax.set_ylabel("Marginal effect")
    ax.set_title(
        f"Marginal effect of {variable}\n"
        "Other selected features held at their means"
    )
    ax.grid(alpha=0.3)

    plt.tight_layout()

    return results

