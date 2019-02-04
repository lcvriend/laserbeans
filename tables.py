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
               ignore_nan=False,
               totals_name='Totals',
               totals_col=True,
               totals_row=True,
               perc_cols=False,
               perc_axis='grand',
               name_abs='abs',
               name_rel='%'):
    """
    Create frequency crosstab for selected categories mapped to specified row and column fields. Group by and count selected categories in df. Then set to rows and columns in crosstab output.

    ---
    :param df: DataFrame
    :param row_fields: str, list (of strings)
        Name(s) of DataFrame field(s) to add into the rows.
    :param column_fields: str, list (of strings)
        Name(s) of DataFrame field(s) to add into the columns.

    Optional keyword arguments:
    ===========================
    :param ignore_nan: boolean, default False
        Ignore category combinations if they have nans.
    :param totals_name: str, default 'Totals'
        Name for total rows/columns (string).
    :param totals_col: boolean, default True
        Add totals column.
    :param totals_row: boolean, default True
        Add totals row.
    :param perc_cols: boolean, default False
        Add relative frequency per column
    :param perc_axis: {'grand', 'index', 'columns'}, or {0,1}, default 'grand'
        'grand' - Calculate percentages from grand total.
        'index', 0 - Calculate percentages from row totals.
        'columns', 1 - Calculate percentages from column totals.
    :param name_abs: str, default 'abs'
        Name for absolute column.
    :param name_rel: str, default '%'
        Name for relative column.
    """

    # assure row and column fields are lists
    if not isinstance(row_fields, list):
        row_fields = [row_fields]
    if not isinstance(column_fields, list):
        column_fields = [column_fields]

    margins = totals_col or totals_row

    # set columns to use/select from df
    group_cols = column_fields.copy()
    group_cols.extend(row_fields)

    if not ignore_nan:
        for col in group_cols:
            if df[col].isnull().values.any():
                if df[col].dtype.name == 'category':
                    df[col] = df[col].cat.add_categories([''])
                df[col] = df[col].fillna('')

    col = df.columns[0]
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

    # remove row/columns where all values are 0
    df = df.loc[(df != 0).any(axis=1)]
    df = df.loc[:, (df != 0).any(axis=0)]
    df = df.astype('int64')

    if perc_cols:
        df = add_perc_cols(df,
                           axis=perc_axis,
                           totals='auto',
                           name_abs=name_abs,
                           name_rel=name_rel)
    return df


def add_perc_cols(df,
                  axis='grand',
                  totals='auto',
                  name_abs='abs',
                  name_rel='%'):
    """
    Add percentage columns for all columns in the DataFrame.

    ---
    :param df: DataFrame

    Optional keyword arguments:
    ===========================
    :param axis: {'grand', 'index', 'columns'}, or {0,1}, default 'grand'
        'grand' - Calculate percentages from grand total.
        'index', 0 - Calculate percentages from row totals.
        'columns', 1 - Calculate percentages from column totals.
    :param totals_row: boolean, {'auto'}, default 'auto'
        'auto' - Check automatically (may backfire).
        True - Take the totals from the DataFrame (last row/column/value).
        False - Calculate the totals.
    :param name_abs: string, default 'abs'
        Name of absolute column.
    :param name_rel: string, default '%'
        Name of relative column.
    """

    nrows, ncols = df.shape

    def check_for_totals_col(df, col, totals_mode):
        total = df.iloc[-1][col]
        if totals_mode == 'auto':
            if not total == df.iloc[:nrows - 1][col].sum():
                total = df[col].sum()
            return total
        if totals_mode:
            return total
        else:
            return df[col].sum()

    def check_for_totals_row(df, totals_mode):
        total = df.iloc[:, -1]
        if totals_mode == 'auto':
            if not total.equals(df.iloc[:, :ncols - 1].sum(axis=1)):
                total = df.sum(axis=1)
            return total
        if totals_mode:
            return total
        else:
            return df.sum(axis=1)

    def check_for_grand_total(df, totals_mode):
        total = df.iloc[-1, -1]
        if totals_mode == 'auto':
            if not total == df.iloc[:nrows - 1, :ncols - 1].values.sum():
                total = check_for_totals_row(df, 'auto').values.sum()
            return total
        elif totals_mode:
            return total
        else:
            return check_for_totals_row(df).values.sum()

    def set_total(df, col, axis, totals):
        maparg = {0: check_for_totals_row,
                  'index': check_for_totals_row,
                  1: check_for_totals_col,
                  'columns': check_for_totals_col,
                  'grand': check_for_grand_total,
                  }
        if not axis in [1, 'columns']:
            total = maparg[axis](df, totals)
        else:
            total = maparg[axis](df, col, totals)
        return total

    df_output = df.copy()
    nlevels = df_output.columns.nlevels + 1
    levels = list(range(nlevels))
    levels.append(levels.pop(0))
    df_output = pd.concat([df_output], axis=1, keys=[name_abs]).reorder_levels(
        levels, axis=1).sort_index(level=[0, 1], ascending=True, axis=1)

    for col in df.columns:
        new_col = col, name_rel
        abs_col = col, name_abs
        if isinstance(col, tuple):
            new_col = *col, name_rel
            abs_col = *col, name_abs

        total = set_total(df, col, axis, totals)
        df_output[new_col] = (df_output[abs_col] / total * 100).round(1)

    levels = list(range(nlevels))
    df_output = df_output.sort_index(level=levels, ascending=True, axis=1)
    df_output = df_output.reindex([name_abs, name_rel], level=nlevels-1, axis=1)

    return df_output


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


def thead(df, tab=4):

    def set_col_names(spans, tab):
        col_names = list()
        for span in spans:
            if span[1] > 1:
                col_names.append(f'{tab * 3}<th class="col_name" colspan="{span[1]}">{span[0]}</th>\n')
            else:
                col_names.append(f'{tab * 3}<th class="col_name">{span[0]}</th>\n')
        return col_names

    tab = ' ' * tab
    html_repr = ''
    all_levels = list()
    nlevels_col = df.columns.nlevels
    nlevels_row = df.index.nlevels
    _, ncols = df.shape

    # column names
    i = 0
    while i < nlevels_col:
        if nlevels_row > 1:
            level = [f'{tab * 3}<th class="col_idx_pre"></th>\n'] * (nlevels_row - 1)
        else:
            level = list()
        col_idx_name = df.columns.get_level_values(i).name
        html_repr = f'{tab * 3}<th class="col_idx_name">{col_idx_name}</th>\n'
        level.append(html_repr)

        col_names = df.columns.get_level_values(i).tolist()
        spans = find_spans(col_names)
        html_repr = set_col_names(spans, tab)
        level.extend(html_repr)

        all_levels.append(level)
        i += 1

    # index names
    idx_names = list(dfx.index.names)
    level = [f'{tab * 3}<td class="row_idx_name">{idx_name}</td>\n' for idx_name in idx_names]
    level.extend([f'{tab * 3}<td class="row_idx_post"></td>\n'] * ncols)
    all_levels.append(level)

    # convert to html
    html = ''
    for level in all_levels:
        html += f'{tab * 2}<tr class="tbl_row">\n'
        html += ''.join(level)
        html += f'{tab * 2}</tr>\n'
    thead = f'{tab}<thead>\n{html}{tab}</thead>\n'
    return thead


def tbody(df, tid=1, tab=4):

    def set_row_names(spans):
        row_names = list()
        for span in spans:
            if span[1] > 1:
                row_names.append(f'<th class="row_name" rowspan="{span[1]}">{span[0]}</th>\n')
            else:
                row_names.append(f'<th class="row_name">{span[0]}</th>\n')
            nones = [None] * (span[1] - 1)
            row_names.extend(nones)
        return row_names

    tab = ' ' * tab
    row_elements = list()

    # indices
    nlevels_row = df.index.nlevels
    i = 0
    while i < nlevels_row:
        idx_names = df.index.get_level_values(i)
        spans = find_spans(idx_names)
        level = set_row_names(spans)
        row_elements.append(level)
        i += 1

    # values
    row_vals = list()
    for row_idx, row in enumerate(df.values):
        val_line = (tab * 3).join([f'<td class="tbl_cell" id="{tid}-{row_idx + 1}|{col_idx + 1}">{item}</td>\n' for col_idx, item in enumerate(row)])
        row_vals.append(val_line)
    row_elements.append(row_vals)

    # zip indices and values
    rows = list(zip(*row_elements))

    # write tbody
    html = ''
    for row in rows:
        row_str = ''
        row_str = (tab * 2) + '<tr class="tbl_row">\n'
        row_str += ''.join([(tab * 3) + item for item in row if item is not None])
        row_str += (tab * 2) + '</tr>\n'
        html += row_str
    tbody = f'{tab}<tbody>\n{html}{tab}</tbody>\n'
    return tbody


def find_spans(idx_vals):
    spans = list()
    for val in idx_vals:
        try:
            if not val == spans[-1][0]:
                spans.append((val, 1))
            else:
                val_tup = spans.pop(-1)
                new_val_tup = val, val_tup[1] + 1
                spans.append(new_val_tup)
        except:
            spans.append((val, 1))
    return spans
