import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

