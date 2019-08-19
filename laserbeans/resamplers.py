"""
resamplers
==========
Functions for resampling data from a DataFrame.
"""

import pandas as pd
import laserbeans.dates as dates


def compare_years(
    df,
    date_field,
    admyear_field,
    start,
    end,
    years=None,
    unit='D'
    ):
    """
    Compares frequencies from `date_field` over multiple administrative years.
    `start` and `end` set the time period to compare in `units`:
    - D: days
    - W: weeks
    - M: months
    If `start` is bigger than `end`, then it is assumed that
    the administrative year starts in the previous calendar year.
    For example:
    > student enrolment for academic year 2018 starts in October 2017.

    Parameters
    ----------
    df: DataFrame
    date_field: str
        Name of date field to resample.
    admyear_field: str
        Name of field containing the administrative year.
    start: int
        Start period unit.
    end: int
        End period unit.
    years: list, optional
        Years to compare.
        If None selects all years in df.
    unit: 'D', 'W', 'M', default: 'D'
        Unit of time to resample the data to.

    Returns
    -------
    DataFrame:
        Outputs a DataFrame where:
        - each column represents an administrative year;
        - each row represents a unit time;
        - each cell represents a frequency.
        Use .cumsum() on output to create cumulative frequencies.
    """

    if start < end:
        span = False
        idx = list(range(start, end))
    else:
        span = True
        idx = list(range(start, dates.UNIT_LENGTH[unit] + 1))
        idx.extend(list(range(1, end)))
    df_output = pd.DataFrame(index=idx)

    if not years:
        years = sorted(list(df[admyear_field].unique()))

    for year in years:
        if not span:
            start_dt = dates.convert_dt[unit](year, start)
        else:
            start_dt = dates.convert_dt[unit](year-1, start)
        end_dt = dates.convert_dt[unit](year, end)

        df_tmp = selector(
            df.query(f"{admyear_field} == @year"),
            date_field,
            start=start_dt,
            end=end_dt
            )
        df_tmp = resample(
            df_tmp,
            admyear_field,
            date_field,
            start=start_dt,
            end=end_dt,
            unit=unit,
            use_dt=False
            )

        df_output = df_output.merge(
            df_tmp, how='left', left_index=True, right_index=True
            )

    return df_output


def resample(
    df, cat_field, date_field, start, end, unit='D', use_dt=True
    ):
    """
    Resample frequencies of categories:
    - time period: from `start` date to `end` date
    - `unit`: days (D), weeks (W), months (M)

    Parameters
    ----------
    df: DataFrame
    cat_field: str
        Name of category field.
    date_field: str
        Name of date field.
    start: datetime
        Start of period as date.
    end: datetime
        End of period as date.
    unit:
        Unit of time to resample to ('D', 'W', 'M').

    Returns
    -------
    DataFrame:
        Outputs a DataFrame where:
        - each column represents a category;
        - each row represents a unit time;
        - each cell represents a frequency.
        Use .cumsum() on output to create cumulative frequencies.
    """

    start = dates.to_dt(start)
    end = dates.to_dt(end)

    categories = df[cat_field].unique()
    date_range = pd.date_range(start, end, freq=unit)
    df_output = pd.DataFrame(index=date_range)
    df_output.index.name = date_field

    for cat in categories:
        df_col = (
            df
            .loc[df[cat_field] == cat, [date_field, cat_field]]
            .set_index(date_field)
            .resample(unit)
            .count()
            .rename(columns={cat_field: cat})
            )
        df_output = df_output.merge(
            df_col, how='left', left_index=True, right_index=True
            )

    if not use_dt:
        df_output.index = getattr(df_output.index, dates.DT_TRANSFORM[unit])

    return df_output.fillna(0).astype(int)


def selector(df, date_field, start, end):
    """
    Select records within a period.

    ---
    :param df: DataFrame.
    :param date_field: Name of date field to query (string)
    :param start: Start of period as date.
    :param end: End of period as date.
    """

    start = dates.to_dt(start)
    end   = dates.to_dt(end)

    qry = (
        f"{date_field} >= @start & "
        f"{date_field} < @end"
    )
    return df.query(qry)
