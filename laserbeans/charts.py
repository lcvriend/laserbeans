"""
charts
======
Wrappers for altair for creating interactive charts from a DataFrame.
"""

import pandas as pd
import altair as alt
import laserbeans.dates_n_periods as dnp


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
        return alt.DateTime(year=dt.year, month=dt.month, date=dt.day)

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
    ilegend = alt.Chart(source, width=30, height=height).mark_square(cursor='pointer').encode(
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


def generate_bin_chart(df, y_name, var_name, var_order=None):
    """
    Generate interactive altair chart from DataFrame:
    - Converts df from wide-form to long-form
    - Transforms index name to lowercase.
    - Creates interactive legend based on number of categories

    ---
    :param df: DataFrame.
    :param y_name: Name for value on the y-axis (string).
    :param var_name: Name of categorized variable (string).
    :param var_order: [Optional] Order in which the category names in var_name should be displayed in the legend (list)
    """

    bin_sort = df.index.tolist()
    x = df.index.name = df.index.name.lower()
    df = df.reset_index().melt(x, var_name=var_name, value_name=y_name)

    if not var_order:
        var_order = df[var_name].unique().tolist()

    height = 30 * len(var_order)
    source = df

    # selections
    multi = alt.selection_multi(fields=[var_name], empty='all')

    # interactive legend
    ilegend = alt.Chart(source, width=30, height=height).mark_square(cursor='pointer').encode(
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
    chart = alt.Chart(source, width=600, mark=alt.MarkDef('bar', clip=True)).encode(
        x=alt.X(f'{x}:O',
                sort=bin_sort
                ),
        y=alt.Y(f'{y_name}:Q',
                stack='zero',
                ),
        color=alt.Color(f'{var_name}:N',
                        legend=None),
        order=alt.Order(f'{var_name}:N', sort='descending'),
    ).transform_filter(
        multi
    )

    # combine
    figure = chart | ilegend

    return figure
