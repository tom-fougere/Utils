import pandas as pd
import numpy as np
from dataframe_computation import *

dict = {'name': ["aparna", "pankaj", "sudhir", "aparna"],
        'degree': ["MBA", "BCA", "M.Tech", "MBA"],
        'score': [90, 40, 80, 98],
        'test': [1, -1.4, np.nan, 0],
        'train': [1, 1, np.inf, 0],
        'valid': [1, np.nan, np.nan, 1]}
df = pd.DataFrame(dict)


def test_get_nb_rows_with_nan():

    nb_rows, percent = get_nb_rows_with_nan(df)

    assert(nb_rows == 2)
    assert(percent == 0.5)


def test_get_stats_from_dataframe():

    df_info = get_stats_from_dataframe(df)

    result_dict = {'Type': [np.dtype('object'), np.dtype('object'), np.dtype('int64'), np.dtype('float64'), np.dtype('float64'), np.dtype('float64')],
                   'Count': [4, 4, 4, 4, 4, 4],
                   'Unique': [3, 3, 4, 3, 2, 1],
                   'Count_nan': [0, 0, 0, 1, 1, 2],
                   'Top': ["aparna", "MBA", 98.0, 0.0, 1.0, 1.0],
                   'Freq': [2, 2, 1, 1, 2, 2],
                   'Mean': [None, None, 77.0, (1-1.4)/3, (1+1)/3, (1+1)/2],
                   'Median': [None, None, 85.0, 0.0, 1.0, 1.0],
                   'Std': [None, None, 25.74231276841043, 1.2055427546683415, 0.5773502691896258, 0.0],
                   'Min': [None, None, 40, -1.4, 0, 1],
                   'Max': [None, None, 98, 1, 1, 1]}
    result = pd.DataFrame(result_dict)
    result.index = df.columns

    pd.testing.assert_frame_equal(df_info, result)


if __name__ == '__main__':
    test_get_nb_rows_with_nan()
    test_get_stats_from_dataframe()
