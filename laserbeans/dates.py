"""
dates_n_periods
===============
Helper functions and constants for dealing with dates.
"""

from datetime import datetime


DATE_FORMATS = [
    '%d-%m-%Y',
    '%d-%m-%y',
    '%Y-%m-%d',
    ]

DT_TRANSFORM = {
    'D': 'dayofyear',
    'W': 'weekofyear',
    'M': 'month',
}

UNIT_LENGTH = {
    'D': 366,
    'W': 52,
    'M': 12,
}


def to_dt(arg):
    """
    Convert input to datetime.
    For convenience this function accepts several types of input:
    - Datetime (returns itself)
    - String (uses DATE_FORMATS with strptime for conversion)
    - Tuple (year, month, day)
    """

    elif isinstance(arg, datetime):
        return arg
    elif isinstance(arg, str):
        for f in DATE_FORMATS:
            try:
                return datetime.strptime(arg, f)
            except ValueError:
                continue
        else:
            raise ValueError(
                f"String '{arg}' does not fit permissable formats:'"
                f"{DATE_FORMATS}"
                )
    else:
        return datetime(*arg)


def correct_date(date, threshold):
    threshold = to_dt(threshold)
    if date < threshold:
        return threshold
    else:
        return date


def dayofyear2date(year, dayofyear):
    return datetime.strptime(f'{year} {dayofyear:03}', '%Y %j')


def weekofyear2date(year, weekofyear):
    return datetime.strptime(f'{year} {weekofyear:02} 1', '%Y %W %w')


def month2date(year, month):
    return datetime(year, month, 1)


convert_dt = {
    'D': dayofyear2date,
    'W': weekofyear2date,
    'M': month2date,
}
