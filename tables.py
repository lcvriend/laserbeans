"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import numpy as np
import pandas as pd


def crosstab_f(df,
               row_fields,
               column_fields,
               totals_name='Totals',
               totals_col=True,
               totals_row=True):
    """
    Create frequency crosstab for selected categories mapped to specified row and column fields. Group by and count selected categories in df. Then set to rows and columns in crosstab output.

    ---
    :param df: DataFrame.
    :param row_fields: Fields to add into the rows (list).
    :param column_fields: Fields to add into the columns (list).

    Optional keyword arguments:    
    :param totals_name: Name for total rows/columns (string) [default='Totals'].
    :param totals_col: Add totals column if True (boolean) [default=True]
    :param totals_row: Add totals row if True (boolean) [default=True]
    """

    if isinstance(row_fields, str):
        row_fields = [row_fields]
    if isinstance(column_fields, str):
        column_fields = [column_fields]

    margins = totals_col or totals_row

    col = df.columns[0]
    group_cols = column_fields.copy()
    group_cols.extend(row_fields)
    df = df.groupby(group_cols)[[col]].count()
    df = df.dropna()
    df = pd.pivot_table(df.reset_index(),
                        index=row_fields,
                        columns=column_fields,
                        aggfunc='sum',
                        dropna=False,
                        margins=margins,
                        margins_name=totals_name)
    df = df.dropna(how='all')
    if margins:
        if not totals_col:
            df = df.drop(totals_name, axis=1, level=1)
        if not totals_row:
            try:
                df = df.drop(totals_name, axis=0, level=0)
            except:
                df = df.drop(totals_name, axis=0)
    df.columns = df.columns.droplevel(0)
    df = df.fillna(0)
    return df.astype(int)


def build_formatters(df, format):
    return {column:format
            for (column, dtype) in df.dtypes.iteritems()
                if dtype in [np.dtype('int32'),
                             np.dtype('int64'),
                             np.dtype('float32'),
                             np.dtype('float64')]}


def table_to_html(df, filename):
    num_format = lambda x: '{:,}'.format(x)
    formatters = build_formatters(df, num_format)
    html_table = df.to_html(formatters=formatters).replace('.0', '').replace(',', '.')
    with open(filename, 'w') as f:
        f.write(html_table)
