def compare_datasets(train_df, test_df):
    """
    Compare all columns in training and test datasets.

    Numeric columns:
        - Mean

    Categorical columns:
        - Percentage of observations in each category

    Returns:
        DataFrame with train, test, and percent_difference columns.
    """

    import pandas as pd
    import numpy as np

    output = {}

    # Include every column from both dataframes
    columns = list(dict.fromkeys(
        train_df.columns.tolist() + test_df.columns.tolist()
    ))

    for column in columns:
        train_series = (
            train_df[column]
            if column in train_df.columns
            else pd.Series(dtype="object")
        )

        test_series = (
            test_df[column]
            if column in test_df.columns
            else pd.Series(dtype="object")
        )

        # Treat the column as numeric if both versions are numeric
        if (
            pd.api.types.is_numeric_dtype(train_series)
            and pd.api.types.is_numeric_dtype(test_series)
        ):
            output[column] = {
                "train": train_series.mean(),
                "test": test_series.mean()
            }

        else:
            # Include all categories found in either dataframe
            categories = pd.Index(
                train_series.dropna().unique().tolist()
                + test_series.dropna().unique().tolist()
            ).unique()

            for category in categories:
                train_percentage = (
                    train_series.eq(category).mean() * 100
                    if len(train_series) > 0
                    else np.nan
                )

                test_percentage = (
                    test_series.eq(category).mean() * 100
                    if len(test_series) > 0
                    else np.nan
                )

                row_name = f"{column} = {category}"

                output[row_name] = {
                    "train": train_percentage,
                    "test": test_percentage
                }

    result = pd.DataFrame.from_dict(output, orient="index")

    # Signed percentage difference relative to training
    result["percent_difference"] = np.where(
        result["train"] != 0,
        ((result["test"] - result["train"]) / result["train"]) * 100,
        np.nan
    )

    result.columns = [
        "train",
        "test",
        "percent_difference"
    ]

    return result

