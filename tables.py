"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import pandas as pd


def crosstab_f(df,
               row_fields,
               column_fields,
               totals_name='Total',
               drop_totals_col=False):
    """
    Create frequency crosstab for given row and column fields.

    ---
    :param df: DataFrame.
    :param row_fields: Fields to add into the rows (list).
    :param column_fields: Fields to add into the columns (list).
    :param totals_name: Name for total rows/columns (string) [default='Total'].
    :param drop_totals_col: Drop totals column if True (boolean) [default='False']
    """

    if isinstance(row_fields, str):
        row_fields = [row_fields]
    if isinstance(column_fields, str):
        column_fields = [column_fields]

    col = df.columns[0]
    group_cols = column_fields.copy()
    group_cols.extend(row_fields)
    df = df.groupby(group_cols)[[col]].count()
    df = df.dropna()
    df = pd.pivot_table(df.reset_index(),
                        index=row_fields,
                        columns=column_fields,
                        aggfunc='sum',
                        margins=True,
                        margins_name=totals_name)
    if drop_totals_col:
        df = df.drop(totals_name, axis=1, level=1)
    df.columns = df.columns.droplevel(0)
    df = df.fillna(0)
    return df.astype(int)
