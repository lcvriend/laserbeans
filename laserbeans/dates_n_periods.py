"""
dates_n_periods
===============
Helper functions and constants for dealing with dates and periods.
"""

import datetime as dt
import pandas as pd


def to_timestamp(arg):
    """
    Convert input to timestamp. For convenience this function accepts several types of input:
    - Timestamp (returns itself)
    - Datetime
    - String (uses strptime for conversion)
    - Tuple (year, month, day)
    """

    if isinstance(arg, pd.Timestamp):
        return arg
    elif isinstance(arg, dt.datetime):
        return pd.Timestamp(arg)
    elif isinstance(arg, str):
        date_formats = [
            '%d-%m-%Y',
            '%d-%m-%y',
            '%Y-%m-%d',
        ]
        timestamp = None
        for date_format in date_formats:
            try:
                timestamp = pd.Timestamp.strptime(arg, date_format)
            except ValueError:
                continue
            else:
                break
        if not isinstance(timestamp, pd.Timestamp):
            raise ValueError(
                f'String {arg} does not fit permissable formats: {[date_format for date_format in date_formats]}')
        else:
            return timestamp
    else:
        return pd.Timestamp(*arg)


def correct_date(date, threshold):
    threshold = to_timestamp(threshold)
    if date < threshold:
        return threshold
    else:
        return date


def dayofyear2date(year, dayofyear):
    return pd.Timestamp.strptime(f'{year} {dayofyear:03}', '%Y %j')


def weekofyear2date(year, weekofyear):
    return pd.Timestamp.strptime(f'{year} {weekofyear:02} 1', '%Y %W %w')


def month2date(year, month):
    return pd.Timestamp(year, month, 1)


def year2date(year):
    return pd.Timestamp(year, 1, 1)


dt_conversion = {
    'D': dayofyear2date,
    'W': weekofyear2date,
    'M': month2date,
    'Y': year2date,
}

DATE_FORMAT = {
    'D': '%d %b %Y',
    'W': 'week: %U, %Y',
    'M': '%b, %Y',
}

DT_TRANSFORM = {
    'D': 'dayofyear',
    'W': 'weekofyear',
    'M': 'month',
    'Y': 'year',
}

UNIT_LENGTH = {
    'D': 355,
    'W': 52,
    'M': 12,
    'Y': 1,
}
