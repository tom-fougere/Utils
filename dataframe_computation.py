import pandas as pd
import numpy as np


def get_nb_rows_with_nan(df):
    """
    Extract the number of rows including NaN values (NaN, Inf)

    Arguments:
    df -- pandas dataframe
    Return:
    nb_rows -- integer corresponding to the number of rows with NaN value(s)
    percent -- scalar corresponding to the percent of rows with NaN in all dataframe (between 0 and 1.)
    """
    data = df.T.isnull().any().value_counts()

    nb_rows = data[1]
    percent = nb_rows / (nb_rows + data[0])

    return nb_rows, percent


def get_stats_from_dataframe(df):
    """
    Display informations and statistics about a dataframe
    Info:
        - Type: Dtype of element
        - Count: number of elements
        - Unique: number of unique elements
        - Count_nan: Number of NaN elements
        - Top: Most common element
        - Freq: Occurrence of the most common element
        - Mean: Mean values (for scalar elements)
        - Median: Median values (for scalar elements)
        - Std: Standard deviation (for scalar elements)
        - Min: Minimal values (for scalar elements)
        - Max: Maximal value (for scalar elements)

    Arguments:
    df -- pandas dataframe
    Return:
    df_info -- pandas dataframe with information
    """

    # New dataframe
    df_info = pd.DataFrame()

    # Replace Inf values by Nan
    df_nan = df.replace([np.inf, -np.inf], np.nan)

    for(columnName, columnData) in df_nan.iteritems():

        count_val = len(columnData)
        unique_val = columnData.nunique()
        count_nan_val = len(columnData) - columnData.count()
        most_common_val = columnData.value_counts().head(1)
        top_val = most_common_val.index.tolist()[0]
        freq_val = most_common_val.values[0]
        type_val = columnData.dtype
        if not np.issubdtype(type_val, np.object):
            mean_val = columnData.mean().astype('float64')
            median_val = columnData.median().astype('float64')
            std_val = columnData.std().astype('float64')
            min_val = columnData.min().astype('float64')
            max_val = columnData.max().astype('float64')
        else:
            mean_val = None
            median_val = None
            std_val = None
            min_val = None
            max_val = None

        df_info = df_info.append({'Type': type_val,
                                  'Count': count_val,
                                  'Unique': unique_val,
                                  'Count_nan': count_nan_val,
                                  'Top': top_val,
                                  'Freq': freq_val,
                                  'Mean': mean_val,
                                  'Median': median_val,
                                  'Std': std_val,
                                  'Min': min_val,
                                  'Max': max_val},
                                 ignore_index=True)

    # Force some types
    df_info = df_info.astype({'Count': 'int64',
                              'Unique': 'int64',
                              'Count_nan': 'int64',
                              'Freq': 'int64'})

    # Re order the columns
    df_info = df_info.reindex(columns=['Type', 'Count', 'Unique', 'Count_nan', 'Top', 'Freq',
                                       'Mean', 'Median', 'Std', 'Min', 'Max'])

    # Change row names
    df_info.index = df.columns

    return df_info
