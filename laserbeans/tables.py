"""
tables
======
Wrappers for pandas for transforming DataFrame into aggregated tables.
"""

import itertools
import pkgutil
import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from pathlib import Path
from IPython.core.display import HTML
import laserbeans.dates_n_periods as dnp
import laserbeans.selectors as sel


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

    Parameters
    ==========
    :param df: DataFrame
    :param row_fields: str, list (of strings)
        Name(s) of DataFrame field(s) to add into the rows.
    :param column_fields: str, list (of strings), None
        Name(s) of DataFrame field(s) to add into the columns.

    Optional keyword arguments
    ==========================
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

    if not column_fields:
        column_fields = '_tmp'
        df[column_fields] = '_tmp'

    # assure row and column fields are lists
    if not isinstance(row_fields, list):
        row_fields = [row_fields]
    if not isinstance(column_fields, list):
        column_fields = [column_fields]

    margins = totals_col or totals_row

    # set columns to use/select from df
    group_cols = column_fields.copy()
    group_cols.extend(row_fields)

    # fill nan if ignore_nan is False
    if not ignore_nan:
        for col in group_cols:
            if df[col].isnull().values.any():
                if df[col].dtype.name == 'category':
                    df[col] = df[col].cat.add_categories([''])
                df[col] = df[col].fillna('')

    # find column for counting that is not in group_cols
    check = False
    i = 0
    while not check:
        try:
            col = df.columns[i]
            if not col in group_cols:
                check = True
            i += 1
        except:
            df['_tmp'] = '_tmp'
            col = '_tmp'
            check = True

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

    # try to remove temp column
    try:
        df = df.drop('_tmp', axis=1)
        df.columns.name = ''
    except:
        pass

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

    Parameters
    ==========
    :param df: DataFrame

    Optional keyword arguments
    ==========================
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
            return check_for_totals_row(df, False).values.sum()

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
        levels, axis=1)

    for col in df.columns:
        new_col = col, name_rel
        abs_col = col, name_abs
        if isinstance(col, tuple):
            new_col = *col, name_rel
            abs_col = *col, name_abs

        total = set_total(df, col, axis, totals)
        col_idx = df_output.columns.get_loc(abs_col)
        new_cols = df_output.columns.insert(col_idx + 1, new_col)
        df_output = pd.DataFrame(df_output, columns=new_cols)
        df_output[new_col] = (df_output[abs_col] / total * 100).round(1)

    return df_output


def crosstab_bin(df, target_field, bin_size, cat_field=None):
    """
    Create crosstab from frequency count of binned observations and (optional) category variable.

    Parameters
    ==========
    :param df: DataFrame
    :param target_field: string
        Name of field to be binned.
    :param bin_size: {float, integer}
        Size of the bins.

    Optional keyword arguments
    ==========================
    :param cat_field: string, default None
        Name of category field.
    """

    grouper = list(filter(None, [cat_field, 'bin']))
    start = df[target_field].min()
    end = df[target_field].max()
    bin_range = pd.interval_range(start=start, end=end, freq=bin_size)

    df['bin'] = pd.cut(df[target_field], bins=bin_range)
    df = df.groupby(grouper)['bin'].count().to_frame()
    df.columns = ['count']

    if df.index.nlevels == 2:
        df = df.unstack(0)
        df.columns = df.columns.droplevel(0)
    df.index = pd.Index([str(bin_).replace(', ', '-') for bin_ in df.index.tolist()], name=target_field)
    return df


def quick_bin(df, target_field, bin_size, bin_col='bin', bin_str=False):
    """
    Add column to DataFrame where values from target field are categorized into bins of bin_size.

    Parameters
    ==========
    :param df: DataFrame
    :param target_field: string
        Name of field to be binned.
    :param bin_size: {float, integer}
        Size of the bins.

    Optional keyword arguments
    ==========================
    :param bin_col: string, default 'bin'
        Name of bin field.
    """

    df = df.copy()

    # define bins according to bin size
    start = df[target_field].min()
    max_ = df[target_field].max()
    nbins = np.ceil((max_ - start) / bin_size)
    end = nbins * bin_size + start
    bin_range = pd.interval_range(start=start,
                                  end=end,
                                  periods=nbins,
                                  closed='left')

    df[bin_col] = pd.cut(df[target_field], bins=bin_range)

    if bin_str:
        # df[bin_col] = df[bin_col].astype(str).str.replace(', ', '-')
        cat_names = [f'[{x.left}-{x.right})'
                     for x in df[bin_col].cat.categories.values]
        cat = CategoricalDtype(categories=cat_names, ordered=True)
        df[bin_col] = df[bin_col].cat.rename_categories(cat_names)
        df[bin_col] = df[bin_col].astype(cat)

    return df


def aggregate_time(df, date_field, start='min', end='max', grouper_cols=None, unit='D', use_dt=False):
    """
    Aggregate records per time unit of date_field and count them.

    Parameters
    ==========
    :param df: DataFrame
    :param date_field: string
        Name of field containing date to be aggregated.

    Optional keyword arguments
    ==========================
    :param start: {'min'}, or {Timestamp, Datetime, date string, tuple (year, month, day)}, default 'min'
        'min' - Use earliest date in date field
        Start date for aggregation period.
    :param end: {'max'}, or {Timestamp, Datetime, date string, tuple (year, month, day)}, default 'max'
        'max' - Use latest date in date field
        End date for aggregation period.
    :param grouper_cols: str, list (of strings), None, default None
        Name(s) of DataFrame field(s) to group by.
    :param unit: {'D', 'W', 'M', 'Y'}, default 'D'
        Unit of time to resample the data to.
        'D' - day of year
        'W' - week of year
        'M' - month of year
        'Y' - year
    :param use_dt: boolean, default False
        Return date instead of a number.
    """
    df = df.copy()

    if start == 'min':
        start = df[date_field].min()
    if end == 'max':
        end = df[date_field].max()
    df = sel.selector(df, date_field, start=start, end=end)

    if not grouper_cols:
        grouper_cols = '_tmp'
        df[grouper_cols] = '_tmp'
    if not isinstance(grouper_cols, list):
        grouper_cols = [grouper_cols]

    grouper_cols.insert(0, unit)

    # find column for counting that is not in group_cols
    check = False
    i = 0
    while not check:
        try:
            col = df.columns[i]
            if not col in grouper_cols:
                check = True
            i += 1
        except:
            df['_tmp'] = '_tmp'
            col = '_tmp'
            check = True

    df[unit] = getattr(df[date_field].dt, dnp.DT_TRANSFORM[unit])

    if use_dt:
        grouper_cols.insert(1, '_year')
        df['_year'] = getattr(df[date_field].dt, dnp.DT_TRANSFORM['Y'])
    df_output = df.groupby(by=grouper_cols)[col].count().unstack()

    if use_dt:
        if df_output.columns.dtype.name == 'category':
            df_output.columns = df_output.columns.add_categories(['_year', unit])
        df_output = df_output.reset_index()

        df_output.index = df_output.apply(lambda row: dnp.dt_conversion[unit](row['_year'].astype(int), row[unit].astype(int)), axis=1)

        df_output = df_output.drop(['_year', unit], axis=1)
        df_output.columns = df_output.columns.remove_unused_categories()

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


class FancyTable:
    path_to_css = Path(__file__).resolve().parent / 'table.css'
    css = path_to_css.read_text()
    tab = 4

    def __init__(self, df):
        self.df = df

    def set_values(self):
        self.nlevels_col = self.df.columns.nlevels
        self.nlevels_row = self.df.index.nlevels
        self.nrows, self.ncols = self.df.shape
        self.col_edges = self.find_edges()

    @staticmethod
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

    def find_edges(self):
        col_edges = list()
        if self.nlevels_col > 1:
            col_names = self.df.columns.get_level_values(-2).tolist()
            spans = self.find_spans(col_names)
            spans = [span[1] for span in spans]
            col_edges = list(itertools.accumulate(spans))
            col_edges = [col_edge - 1 for col_edge in col_edges]
        return col_edges

    def html(self):
        self.set_values()
        html_tbl = f'{self.css}<table class="laserbeans">\n{self.thead()}{self.tbody()}</table>\n'
        return html_tbl

    def display(self):
        display(HTML(self.html()))

    def thead(self):
        tab = ' ' * self.tab
        html_repr = ''
        all_levels = list()

        # column index names
        def set_col_names(spans, tab, level):
            col_names = list()
            if i == (self.nlevels_col - 1):
                for spn_idx, span in enumerate(spans):
                    html_repr = f'{tab * 3}<th class="col_name" colspan="{span[1]}">{span[0]}</th>\n'
                    if spn_idx in self.col_edges:
                        html_repr = f'{tab * 3}<th class="col_name col_edge" colspan="{span[1]}">{span[0]}</th>\n'
                    col_names.append(html_repr)
            else:
                for span in spans:
                    if span[1] > 1:
                        html_repr = f'{tab * 3}<th class="col_name col_edge" colspan="{span[1]}">{span[0]}</th>\n'
                        col_names.append(html_repr)
                    else:
                        html_repr = f'{tab * 3}<th class="col_name col_edge">{span[0]}</th>\n'
                        col_names.append(html_repr)
            return col_names

        i = 0
        while i < self.nlevels_col:
            level = list()

            # column index
            col_idx_name = self.df.columns.get_level_values(i).name
            if col_idx_name == None:
                col_idx_name = ''

            if self.nlevels_row > 1:
                html_repr = f'{tab * 3}<th class="col_idx_name" colspan="{self.nlevels_row}">{col_idx_name}</th>\n'
            else:
                html_repr = f'{tab * 3}<th class="col_idx_name">{col_idx_name}</th>\n'
            level.append(html_repr)

            # column names
            col_names = self.df.columns.get_level_values(i).tolist()
            spans = self.find_spans(col_names)
            html_repr = set_col_names(spans, tab, i)
            level.extend(html_repr)

            all_levels.append(level)
            i += 1

        # row index names
        def html_repr_idx_names(idx_name):
            html_repr = f'{tab * 3}<td class="row_idx_name">{idx_name}</td>\n'
            return html_repr

        idx_names = list(self.df.index.names)
        level = [html_repr_idx_names(idx_name) for idx_name in idx_names]

        def html_repr_idx_post(col_idx, item):
            html_repr = f'{tab * 3}<td class="row_idx_post"></td>\n'
            if col_idx in self.col_edges:
                html_repr = f'{tab * 3}<td class="row_idx_post col_edge"></td>\n'
            return html_repr

        level.extend([html_repr_idx_post(col_idx, item) for col_idx, item in enumerate([''] * self.ncols)])
        all_levels.append(level)

        # convert to html
        html = ''
        for level in all_levels:
            html += f'{tab * 2}<tr class="tbl_row">\n'
            html += ''.join(level)
            html += f'{tab * 2}</tr>\n'
        thead = f'{tab}<thead>\n{html}{tab}</thead>\n'
        return thead

    def tbody(self, tid='cell'):

        tab = ' ' * self.tab
        row_elements = list()

        # indices
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

        i = 0
        while i < self.nlevels_row:
            idx_names = self.df.index.get_level_values(i)
            spans = self.find_spans(idx_names)
            level = set_row_names(spans)
            row_elements.append(level)
            i += 1

        # values
        def html_repr(col_idx, item):
            html_repr = f'<td id="{tid}-{row_idx + 1}-{col_idx + 1}" class="tbl_cell">{item}</td>\n'
            if col_idx in self.col_edges:
                html_repr = f'<td id="{tid}-{row_idx + 1}-{col_idx + 1}" class="tbl_cell col_edge">{item}</td>\n'
            return html_repr

        values = self.df.astype(str).values # cast all values as strings
        row_vals = list()
        for row_idx, row in enumerate(values):
            val_line = [html_repr(col_idx, item) for col_idx, item in enumerate(row)]
            val_line = (tab * 3).join(val_line)
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
