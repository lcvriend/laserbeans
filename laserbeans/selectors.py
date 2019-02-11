"""
selectors
=========
Helper functions for selecting data from a DataFrame.
"""

import pandas as pd
import laserbeans.dates_n_periods as dnp


def select_years(df, date_field, admyear_field, start, end, years='all', unit='D'):
    """
    Select records within period over multiple administrative years.

    ---
    :param df: DataFrame.
    :param years: Years to compare (list) or 'all' to select all years in df [default = 'all'].
    :param date_field: Date field to select on (string).
    :param admyear_field: Field containing the administrative year (string).
    :param start: Start period unit.
    :param end: End period unit.
    :param unit: Unit of time for selecting the period ('D', 'W', 'M', 'Y') [default = 'D'].
    """

    if years == 'all':
        years = df[admyear_field].unique().tolist()

    if start < end:
        span = False
        idx = list(range(start, end))
    else:
        span = True
        idx = list(range(start, dnp.UNIT_LENGTH[unit] + 1))
        idx.extend(list(range(1, end)))
    df_output = pd.DataFrame(columns=df.columns).set_index(date_field)

    for year in years:
        if not span:
            start_year = dnp.dt_conversion[unit](year, start)
        else:
            start_year = dnp.dt_conversion[unit](year-1, start)
        end_year = dnp.dt_conversion[unit](year, end)

        df_tmp = selector(df.query(f'{admyear_field} == @year'),
                          date_field,
                          start=start_year,
                          end=end_year).set_index(date_field).sort_index()

        df_output = df_output.append(df_tmp, sort=False)

    return df_output


def selector(df, date_field, start, end):
    """
    Select records within a period.

    ---
    :param df: DataFrame.
    :param date_field: Name of date field to query (string)
    :param start: Start of period as date.
    :param end: End of period as date.
    """

    start = dnp.to_timestamp(start)
    end = dnp.to_timestamp(end)

    qry = (
        f"{date_field} >= @start & "
        f"{date_field} < @end"
    )
    return df.query(qry)
