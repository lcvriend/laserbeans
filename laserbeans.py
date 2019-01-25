"""
laserbeans
==========
Toolbox for data exploration.
"""

import datetime as dt
import pandas as pd
import altair as alt
import dates_n_periods as dnp


def compare_years(df, date_field, admyear_field, years, start, end, unit='D'):
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
        delta = False
        idx = list(range(start, end))
    else:
        delta = True

        idx = list(range(start, dnp.UNIT_LENGTH[unit] + 1))
        idx.extend(list(range(1, end)))

    df_output = pd.DataFrame(index=idx)

    for year in years:
        if not delta:
            start_year = dnp.dt_conversion[unit](year, start)
        else:
            start_year = dnp.dt_conversion[unit](year-1, start)
        end_year = dnp.dt_conversion[unit](year, end)

        df_tmp = selector(df.query(f'{admyear_field} == @year'),
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


def selector(df, date_field, start, end):
    """
    Select period within df.

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


def generate_chart(df, y_name, var_name,
                   var_order=None,
                   mark='area',
                   stack=None,
                   zoom=False,
                   unit='M'):
    """
    Generate interactive altair chart from DataFrame:
    - Converts df from wide-form to long-form
    - Transforms index name to lowercase.
    - Sets domain based on min x and max.
    - Formats x based on type (datetime or int)
    - Creates interactive legend based on number of categories

    ---
    :param df: DataFrame.
    :param y_name: Name for value on the y-axis (string).
    :param var_name: Name of categorized variable (string).
    :param var_order: [Optional] Order in which the category names in var_name should be displayed in the legend (list)
    :param mark: Mark type. One of 'bar', 'circle', 'square', 'tick', 'line', 'area', 'point', 'rule', and 'text' (string) [default = 'area'].
    :param stack: Option to stack chart. One of 'zero', 'normalize', 'center' or None [default = None].
    :param zoom: Option to add vertical zoom to chart. One of True or False [default = False].
    """
    def to_altair_datetime(dt):
        dt = pd.to_datetime(dt)
        return alt.DateTime(year=dt.year, month=dt.month, date=dt.day,
                            hours=dt.hour, minutes=dt.minute, seconds=dt.second,
                            milliseconds=0.001 * dt.microsecond)

    x = df.index.name = df.index.name.lower()
    if isinstance(df.index, pd.DatetimeIndex):
        altx = x + ':T'
        xmin = to_altair_datetime(df.index.min())
        xmax = to_altair_datetime(df.index.max())
        formatx = dnp.DATE_FORMAT[unit]
        ticks = [to_altair_datetime(date) for date in pd.date_range(
            df.index.min(), df.index.max(), freq=unit).tolist()]
    else:
        altx = x + ':Q'
        xmin = df.index.min()
        xmax = df.index.max()
        formatx = 'd'
        ticks = list(range(xmin, xmax))
    domainx = (xmin, xmax)

    df = df.reset_index().melt(x, var_name=var_name, value_name=y_name)

    if not var_order:
        var_order = df[var_name].unique().tolist()

    height = 30 * len(var_order)
    source = df

    # selections
    multi = alt.selection_multi(fields=[var_name], empty='all')
    brushx = alt.selection_interval(encodings=['x'])
    if zoom:
        x = None
        domainx = brushx.ref()

    # interactive legend
    ilegend = alt.Chart(source, width=30, height=height).mark_square().encode(
        y=alt.Y(f'{var_name}:N',
                axis=alt.Axis(title=var_name,
                              ticks=False,
                              labelPadding=5,
                              domain=False
                              ),
                sort=alt.Sort(var_order),
                ),
        size=alt.condition(multi,
                           alt.value(200),
                           alt.value(100),
                           ),
        color=alt.condition(multi,
                            alt.Color(f'{var_name}:N', legend=None),
                            alt.value('lightgray'),
                            ),
    ).properties(selection=multi)

    # chart
    chart = alt.Chart(source, width=600, mark=alt.MarkDef(mark, clip=True)).encode(
        x=alt.X(altx,
                scale=alt.Scale(domain=domainx, nice=False),
                axis=alt.Axis(title=x, format=formatx, values=ticks),
                ),
        y=alt.Y(f'{y_name}:Q',
                stack=stack,
                ),
        color=alt.Color(f'{var_name}:N',
                        legend=None),
        order=alt.Order(f'{var_name}:N', sort='descending'),
    ).transform_filter(
        multi
    )

    # zoom
    zoomview = chart.encode(
        x=alt.X(altx,
                scale=alt.Scale(nice=False),
                axis=alt.Axis(format=formatx),
                ),
        y=alt.Y(f'{y_name}:Q',
                axis=alt.Axis(title=None),
                ),
    ).properties(
        height=60
    ).add_selection(brushx)

    # combine
    if zoom:
        figure = ((chart & zoomview) | ilegend)
    else:
        figure = chart | ilegend

    return figure
