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


def add_perc_cols(df, totals_row='auto'):
    """
    Add percentage columns for all columns in the DataFrame.

    ---
    :param df: DataFrame.

    Optional keyword arguments:
    :param totals_row:
        'auto' - Check automatically (may backfire). [default]
        True - Use the last row as a totals row.
        False - Calculate the total.
    """

    def check_for_totals_row(df, col):
        total = df.iloc[-1][col]
        if not total == df.iloc[:len(df) - 1][col].sum():
            total = df[col].sum()
        return total

    def set_total(df, col, totals_row):
        if totals_row == 'auto':
            return check_for_totals_row(df, col)
        elif totals_row:
            return df.iloc[-1][col]
        else:
            return df[col].sum()

    df_output = df.copy()
    levels = list(range(df_output.columns.nlevels + 1))
    levels.append(levels.pop(0))
    df_output = pd.concat([df_output], axis=1, keys=['abs']).reorder_levels(levels, axis=1).sort_index(level=[0, 1], ascending=True, axis=1)

    for col in df.columns:
        new_col = col, '%'
        abs_col = col, 'abs'
        if isinstance(col, tuple):
            new_col = *col, '%'
            abs_col = *col, 'abs'

        total = set_total(df, col, totals_row)
        df_output[new_col] = (df_output[abs_col] / total * 100).round(1)

    levels = list(range(df_output.columns.nlevels))
    sort = [bool(level) for level in levels]
    sort.append(sort.pop(0))

    return df_output.sort_index(level=levels, ascending=sort, axis=1)


def build_formatters(df, format):
    return {column: format
            for (column, dtype) in df.dtypes.iteritems()
            if dtype in [np.dtype('int32'),
                         np.dtype('int64'),
                         np.dtype('float32'),
                         np.dtype('float64')]}


def table_to_html(df, filename):
    def num_format(x): return '{:,}'.format(x)
    formatters = build_formatters(df, num_format)
    html_table = df.to_html(formatters=formatters).replace('.0', '').replace(',', '.')
    with open(filename, 'w') as f:
        f.write(html_table)
