def roc_plot(
    df_1,
    df_2,
    models,
    target_name
):
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True
    )

    for ax, df, title in zip(
        axes,
        [df_1, df_2],
        ["Train", "Test"]
    ):
        y_true = df[target_name]

        for model_name, probability_column in models.items():
            probabilities = df[probability_column]

            fpr, tpr, _ = roc_curve(
                y_true,
                probabilities
            )

            auc_score = roc_auc_score(
                y_true,
                probabilities
            )

            ax.plot(
                fpr,
                tpr,
                linewidth=2,
                label=f"{model_name}: AUC = {auc_score:.3f}"
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color="gray",
            label="Random classifier"
        )

        ax.set_title(title)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    fig.suptitle("ROC Curves by Dataset", fontsize=15)
    plt.tight_layout()
    plt.show()

def cumulative_event_rate_plot(
    df_1,
    df_2,
    models,
    target_name,
    n_bins=10
):
    import matplotlib.pyplot as plt
    import pandas as pd

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6),
        sharex=True,
        sharey=True
    )

    for ax, df, title in zip(
        axes,
        [df_1, df_2],
        ["Train", "Test"]
    ):
        y_true = df[target_name]

        for model_name, probability_column in models.items():
            temp = pd.DataFrame({
                "target": y_true,
                "prediction": df[probability_column]
            }).dropna()

            # Rank highest predictions first
            temp["prediction_rank"] = (
                temp["prediction"]
                .rank(method="first", ascending=False)
            )

            # Bin 1 contains the highest-risk observations
            temp["bin"] = pd.qcut(
                temp["prediction_rank"],
                q=n_bins,
                labels=False
            ) + 1

            # Calculate bads captured in each bin
            bads_by_bin = (
                temp
                .groupby("bin", observed=False)["target"]
                .sum()
                .sort_index()
            )

            # Cumulative number of bads captured
            cumulative_bads = bads_by_bin.cumsum()

            # Cumulative percentage of all bads captured
            cumulative_event_rate = (
                cumulative_bads / temp["target"].sum()
            )

            ax.plot(
                cumulative_event_rate.index,
                cumulative_event_rate.values * 100,
                marker="o",
                linewidth=2,
                label=model_name
            )

        ax.set_title(title)
        ax.set_xlabel(
            f"Prediction bin\n"
            f"(1 = highest predicted risk, {n_bins} = lowest)"
        )
        ax.set_ylabel("Cumulative bads captured (%)")
        ax.set_xticks(range(1, n_bins + 1))
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    fig.suptitle(
        "Cumulative Event Capture by Prediction Bin",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()

def ks_plot(
    df_1,
    df_2,
    models,
    target_name,
    n_bins=10
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    datasets = [
        (df_1, "Train"),
        (df_2, "Test")
    ]

    n_models = len(models)

    fig, axes = plt.subplots(
        n_models,
        2,
        figsize=(14, 5 * n_models),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    for row, (model_name, probability_column) in enumerate(models.items()):

        for col, (df, dataset_name) in enumerate(datasets):
            ax = axes[row, col]

            temp = pd.DataFrame({
                "target": df[target_name],
                "prediction": df[probability_column]
            }).dropna()

            # Highest predicted risk is in bin 1
            temp["prediction_rank"] = (
                temp["prediction"]
                .rank(method="first", ascending=False)
            )

            temp["bin"] = pd.qcut(
                temp["prediction_rank"],
                q=n_bins,
                labels=False
            ) + 1

            distribution_by_bin = (
                temp
                .groupby("bin", observed=False)["target"]
                .agg(
                    bads="sum",
                    observations="count"
                )
                .sort_index()
            )

            distribution_by_bin["goods"] = (
                distribution_by_bin["observations"]
                - distribution_by_bin["bads"]
            )

            cumulative_bad_rate = (
                distribution_by_bin["bads"].cumsum()
                / distribution_by_bin["bads"].sum()
            )

            cumulative_good_rate = (
                distribution_by_bin["goods"].cumsum()
                / distribution_by_bin["goods"].sum()
            )

            ks_by_bin = (
                cumulative_bad_rate
                - cumulative_good_rate
            )

            ks_bin = ks_by_bin.abs().idxmax()
            ks_value = ks_by_bin.abs().loc[ks_bin]

            # Plot cumulative bads
            ax.plot(
                cumulative_bad_rate.index,
                cumulative_bad_rate * 100,
                marker="o",
                linewidth=2,
                label="Cumulative bads"
            )

            # Plot cumulative goods
            ax.plot(
                cumulative_good_rate.index,
                cumulative_good_rate * 100,
                marker="o",
                linewidth=2,
                linestyle="--",
                label="Cumulative goods"
            )

            # Mark KS points
            ax.vlines(
                x=ks_bin,
                ymin=cumulative_good_rate.loc[ks_bin] * 100,
                ymax=cumulative_bad_rate.loc[ks_bin] * 100,
                color="red",
                linewidth=2,
                label=f"KS = {ks_value:.3f}"
            )

            ax.scatter(
                ks_bin,
                cumulative_bad_rate.loc[ks_bin] * 100,
                color="red",
                zorder=5
            )

            ax.scatter(
                ks_bin,
                cumulative_good_rate.loc[ks_bin] * 100,
                color="red",
                zorder=5
            )

            ax.set_title(f"{model_name} — {dataset_name}")
            ax.set_xlabel(
                f"Prediction bin\n"
                f"(1 = highest predicted risk, {n_bins} = lowest)"
            )
            ax.set_ylabel("Cumulative population (%)")
            ax.set_xticks(range(1, n_bins + 1))
            ax.set_ylim(0, 105)
            ax.grid(alpha=0.3)
            ax.legend(loc="best")

    fig.suptitle(
        "KS Plots by Model and Dataset",
        fontsize=16
    )

    plt.tight_layout()
    plt.show()

def calibration_plot(
    df_1,
    df_2,
    models,
    title_1="Train",
    title_2="Test",
    target_name="bad",
    n_bins=10
):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.calibration import calibration_curve

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    for ax, df, title in zip(
        axes,
        [df_1, df_2],
        [title_1, title_2]
    ):
        y_true = df[target_name]

        all_x_values = []
        all_y_values = []

        for model_name, probability_column in models.items():
            predicted_probability = df[probability_column]

            observed_probability, mean_predicted_probability = (
                calibration_curve(
                    y_true,
                    predicted_probability,
                    n_bins=n_bins,
                    strategy="quantile"
                )
            )

            all_x_values.extend(mean_predicted_probability)
            all_y_values.extend(observed_probability)

            ax.plot(
                mean_predicted_probability,
                observed_probability,
                marker="o",
                linewidth=2,
                label=model_name
            )

        all_values = np.concatenate([
            np.asarray(all_x_values),
            np.asarray(all_y_values)
        ])

        value_min = all_values.min()
        value_max = all_values.max()

        data_range = value_max - value_min
        padding = 0.01 if data_range == 0 else data_range * 0.05

        plot_min = max(0, value_min - padding)
        plot_max = min(1, value_max + padding)

        ax.plot(
            [plot_min, plot_max],
            [plot_min, plot_max],
            linestyle="--",
            color="gray",
            label="Perfect calibration"
        )

        ax.set_xlim(plot_min, plot_max)
        ax.set_ylim(plot_min, plot_max)

        ax.set_title(title)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed event probability")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left")

    fig.suptitle(
        "Predicted vs Observed Event Probabilities",
        fontsize=15
    )

    plt.tight_layout()
    plt.show()

