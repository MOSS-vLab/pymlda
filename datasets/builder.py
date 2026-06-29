import os
import pandas as pd

from pymlda.features import extract_features_windowed


def build_dataset_from_folder(
    folder_path,
    fs,
    window_size,
    overlap=0.5,
    domain="all",
    axis_columns=None
):
    """
    Build ML dataset from CSV folder structure.

    Parameters
    ----------
    folder_path : str
        Root folder containing CSV files

    fs : float
        Sampling frequency

    window_size : int
        Window size

    overlap : float
        Window overlap

    domain : str
        Feature domain

    axis_columns : dict
        Dictionary mapping axis names to CSV columns

    Returns
    -------
    dataset : pandas.DataFrame
    """

    if axis_columns is None:

        axis_columns = {
            "X": 1,
            "Y": 2,
            "Z": 3
        }

    dataset = []

    for root, dirs, files in os.walk(folder_path):

        for file in files:

            if file.endswith(".csv"):

                csv_path = os.path.join(root, file)

                try:

                    df = pd.read_csv(csv_path)

                    for axis_name, col_idx in axis_columns.items():

                        if col_idx >= len(df.columns):
                            continue

                        signal = df.iloc[:, col_idx].dropna().values

                        if len(signal) < window_size:
                            continue

                        features_df = extract_features_windowed(
                            signal,
                            fs=fs,
                            window_size=window_size,
                            overlap=overlap,
                            domain=domain
                        )

                        features_df["axis"] = axis_name
                        features_df["file"] = file
                        features_df["folder"] = os.path.basename(root)

                        dataset.append(features_df)

                except Exception as e:

                    print(f"Error processing {file}: {e}")

    if len(dataset) == 0:

        return pd.DataFrame()

    dataset = pd.concat(dataset, ignore_index=True)

    return dataset