"""
resamplers
==========
Functions for resampling data from a DataFrame.
"""

import pandas as pd
import laserbeans.selectors as sel
import laserbeans.dates_n_periods as dnp


def compare_f_years(df, date_field, admyear_field, years, start, end, unit='D'):
    """
    Compare frequencies from `date_field` over multiple administrative years. `Start` and `end` set the time period to compare in `unit`: days (D), weeks (W), months (M) or year (Y). If `start` is bigger than `end` then it is assumed that the administrative year starts in the previous calendar year. For example, student enrolment for academic year 2018 starts in October 2017.

    Output a DataFrame where:
    - each column represents an administrative year;
    - each row represents a unit time;
    - each cell represents a frequency.
    Use .cumsum() on output to create cumulative frequencies.

    ---
    :param df: DataFrame.
    :param years: Years to compare (list).
    :param date_field: Date field to resample.
    :param admyear_field: Field containing the administrative year.
    :param start: Start period unit.
    :param end: End period unit.
    :param unit: Unit of time to resample the data to ('D', 'W', 'M', 'Y').
    """

    if start < end:
        span = False
        idx = list(range(start, end))
    else:
        span = True
        idx = list(range(start, dnp.UNIT_LENGTH[unit] + 1))
        idx.extend(list(range(1, end)))
    df_output = pd.DataFrame(index=idx)

    for year in years:
        if not span:
            start_year = dnp.dt_conversion[unit](year, start)
        else:
            start_year = dnp.dt_conversion[unit](year-1, start)
        end_year = dnp.dt_conversion[unit](year, end)

        df_tmp = sel.selector(df.query(f'{admyear_field} == @year'),
                              date_field,
                              start=start_year,
                              end=end_year)
        df_tmp = resample(df_tmp,
                          admyear_field,
                          date_field,
                          start=start_year,
                          end=end_year,
                          unit=unit,
                          use_dt=False)

        df_output = pd.merge(df_output, df_tmp, how='left', left_index=True, right_index=True)

    return df_output


def resample(df, cat_field, date_field, start, end, unit='D', use_dt=True):
    """
    Resample frequencies of categories within a time period (from `start` date to `end` date) in `unit`: days (D), weeks (W), months (M), year (Y).

    ---
    :param df: DataFrame.
    :param cat_field: Name of category field in df (string).
    :param date_field: Name of date field in df (string).
    :param start: Start of period as date.
    :param end: End of period as date.
    :param unit: Unit of time to resample to ('D', 'W', 'M', 'Y').
    """
    start = dnp.to_timestamp(start)
    end = dnp.to_timestamp(end)

    categories = df[cat_field].unique()
    dates = pd.date_range(start, end, freq=unit)
    df_output = pd.DataFrame(index=dates)
    df_output.index.name = date_field

    for cat in categories:
        df_col = df.loc[:, [date_field, cat_field]]
        df_col[cat] = 1
        qry = f"{cat_field} == @cat"
        df_col = df_col.query(qry)
        df_col = df_col.set_index(date_field)
        df_col = df_col[cat].resample(unit).sum().to_frame()

        df_output = pd.merge(df_output, df_col, how='left', left_index=True, right_index=True)

    df_output = df_output.fillna(method='ffill')
    if not use_dt:
        df_output.index = getattr(df_output.index, dnp.DT_TRANSFORM[unit])

    return df_output
