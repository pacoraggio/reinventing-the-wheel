import numpy as np


def get_outliers(df, col):
    """
    Detect outliers in a DataFrame column using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
    col : str
        Column name to compute outliers on.

    Returns
    -------
    outlier_df : pd.DataFrame
        Subset of df containing only the outlier rows.
    outlier_arr : np.ndarray
        Array of outlier values from the specified column.
    """
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    mask = (df[col] < lower) | (df[col] > upper)
    outlier_df = df[mask].copy()
    outlier_arr = outlier_df[col].to_numpy()

    return outlier_df, outlier_arr