"""Do things to SFL data DataFrames"""
from collections import OrderedDict
import datetime
import json
import os
import re
import pandas as pd
from . import db
from . import errors as sfperrors
from . import geo
from . import util
from . import seaflowfile


sfl_delim = '\t'

# Mappings between SFL file and SQL table column names
colname_mapping = {
    "table_to_file": {
        "file": "FILE",
        "date": "DATE",
        "file_duration": "FILE DURATION",
        "lat": "LAT",
        "lon": "LON",
        "conductivity": "CONDUCTIVITY",
        "salinity": "SALINITY",
        "ocean_tmp": "OCEAN TEMP",  # not a typo
        "par": "PAR",
        "bulk_red": "BULK RED",
        "stream_pressure": "STREAM PRESSURE",
        "flow_rate": "FLOW RATE",
        "event_rate": "EVENT RATE"
    }
}
# Reverse mappings from SQL table to file column headers
colname_mapping["file_to_table"] = {v:k for k, v in colname_mapping["table_to_file"].items()}

# Numeric columns using SQL table column names
numeric_columns = [
    "file_duration", "lat", "lon", "conductivity", "salinity", "ocean_tmp", "par", "bulk_red",
    "stream_pressure", "event_rate"
]

output_columns = [
    "file", "date", "file_duration", "lat", "lon", "conductivity",
    "salinity", "ocean_tmp", "par", "bulk_red", "stream_pressure",
    "event_rate"
]

# numeric min/max values
min_file_duration = 0
min_lat, max_lat = -90, 90
min_lon, max_lon = -180, 180
min_stream_pressure = 1e-4
min_event_rate = 0

def check(df):
    """Perform checks on SFL dataframe

    Returns a list of errors.
    """
    errors = []
    errors.extend(check_numeric(df, "file_duration", minval=min_file_duration, require_all=True))
    errors.extend(check_numeric(df, "lat", minval=min_lat, maxval=max_lat, require_all=True))
    errors.extend(check_numeric(df, "lon", minval=min_lon, maxval=max_lon, require_all=True))
    errors.extend(check_numeric(df, "conductivity", require_all=False))
    errors.extend(check_numeric(df, "salinity", require_all=False))
    errors.extend(check_numeric(df, "ocean_tmp", require_all=False))
    errors.extend(check_numeric(df, "par", require_all=False))
    errors.extend(check_numeric(df, "bulk_red", require_all=False))
    errors.extend(check_numeric(df, "stream_pressure", minval=min_stream_pressure, require_all=True))
    errors.extend(check_numeric(df, "event_rate", minval=min_event_rate, require_all=True))
    errors.extend(check_file(df))
    errors.extend(check_date(df))
    return errors


def check_date(df):
    errors = []
    if "date" not in df.columns:
        errors.append(create_error(df, "date", msg="date column is missing"))
    else:
        # All dates must match RFC 3339 with no fractional seconds
        # only integer seconds.
        date_flags = df["date"].map(check_date_string)
        if len(date_flags) > 0:
            # select rows that failed date check
            for i, v in df[~date_flags]["date"].iteritems():
                errors.append(create_error(df, "date", msg="Invalid date format", row=i, val=v))
    return errors


def check_date_string(date):
    """Confirm value is an RFC3339 string with UTC timezone as [+-]00:00"""
    passed = False
    try:
        dt = datetime.datetime.fromisoformat(date)
    except ValueError:
        pass
    else:
        if dt.tzinfo == datetime.timezone.utc and (date.endswith('+00:00') or date.endswith('-00:00')):
            passed = True
    # Return true if any format is correct
    return passed


def check_file(df):
    errors = []
    # File field must be present
    if "file" not in df.columns:
        errors.append(create_error(df, "file", msg="file column is missing"))
    else:
        # File field must contain well formatted file strings, valid dates, and
        # day-of-year directory.
        def parse_filename(f):
            try:
                s = seaflowfile.SeaFlowFile(f)
            except sfperrors.FileError:
                return False
            if s.path_dayofyear:
                return True
            return False

        good_files_selector = df["file"].map(parse_filename)
        good_files = df[good_files_selector]
        bad_files = df[~good_files_selector]
        for i, v in bad_files["file"].iteritems():
            errors.append(create_error(bad_files, "file", msg="Invalid file name", row=i, val=v))

        # Files must be unique
        dup_files = df.loc[df["file"].duplicated(keep=False), "file"]
        for i, v in dup_files.iteritems():
            errors.append(create_error(dup_files, "file", msg="Duplicate file", row=i, val=v))

        # Files should be in order
        # Only consider files that are parseable by seaflowfile.SeaFlowFile
        inorder = seaflowfile.sorted_files(good_files["file"])
        files_equal = good_files["file"] == inorder
        if not files_equal.all():
            i = int(good_files[~files_equal].index[0])
            v = "First out of order file {}".format(good_files.loc[i, "file"])
            errors.append(create_error(good_files, "file", msg="Files out of order", row=i, val=v))

        # Files should match date in same row
        if "date" in df.columns:
            for i, v in good_files["file"].iteritems():
                s = seaflowfile.SeaFlowFile(v)
                d = good_files.loc[i, "date"]
                if s.is_new_style and s.rfc3339 != d:
                    errors.append(create_error(good_files, "file/date", msg="File and date don't match", row=i, val=f"{v} {d}"))

    return errors


def check_numeric(df, colname, require_all=False, minval=None, maxval=None):
    """
    Check a numeric column.

    Checks that the column is present and has valid numeric values. Optionally
    checks if values are in acceptable range and if all rows have non-NA values.

    Parameters
    -----------
    df: pandas.DataFrame
        SFL DataFrame with column to check.
    colname: str
        Name of the column to check.
    require_all: bool, default False
        Require that all rows have non-NA value?
    minval: int or float, optional
        Minimum acceptable value.
    maxval: int or float, optional
        Maximum acceptable value.

    Returns
    -------
    list of error dicts
        Errors created with create_error()
    """
    errors = []
    if colname not in df.columns:
        # column must be present
        errors.append(create_error(df, colname, msg=f"{colname} column is missing", level="error"))
    else:
        notnas = df[df[colname].notna()]
        numbers = pd.to_numeric(notnas[colname], errors="coerce")
        # Create boolean index for values in acceptable range
        # Start by selecting everything, then select by minval/maxval
        good_selector = pd.np.ones(len(numbers), dtype=bool)
        if minval is not None:
            good_selector = good_selector & (numbers >= minval)
        if maxval is not None:
            good_selector = good_selector & (numbers <= maxval)
        # Catch values outside correct range
        # Catch non-numeric values (NAs created during to_numeric())
        bad_numbers = notnas.loc[~good_selector, colname]
        for i, v in bad_numbers.iteritems():
            errors.append(create_error(df, colname, msg=f"Invalid {colname}", row=i, val=v, level="error"))

        nas = df.loc[df[colname].isna(), colname]
        if require_all:
            if len(nas) > 0:
                # Can't have NAs with require_all
                for i, v in nas.iteritems():
                    errors.append(create_error(df, colname, msg="Missing required data", row=i, val=v, level="error"))
        elif len(nas) == len(df):
            # Warn if no data in column and not require_all
            errors.append(create_error(df, colname, msg=f"{colname} column has no data", level="warning"))

    return errors


def convert_gga2dd(df):
    """Return a copy of df with coordinates converted from GGA to decimal degrees."""
    newdf = df.copy(deep=True)
    newdf["lat"] = df["lat"].map(geo.ggalat2dd, na_action="ignore")
    newdf["lon"] = df["lon"].map(geo.ggalon2dd, na_action="ignore")
    return newdf


def create_error(df, col, msg, row=None, val=None, level='error'):
    """Create an error dictionary.

    Error levels can be 'error' (fatal) or 'warning'.
    """
    level_values = ['error', 'warning']
    if level not in level_values:
        raise ValueError(f"valid values for 'level' are {level_values}")

    e = {
        "column": col,
        "message": msg,
        "line (1-based)": None,
        "value": None,
        "level": level,
    }
    if row is not None:
        e["line (1-based)"] = row + 2
    if val is not None:
        e["value"] = make_json_serializable(val)
    elif row is not None:
        e["value"] = make_json_serializable(df.loc[row, col])
    return e


def dedup(df):
    """Remove duplicate rows from DataFrame by "file".

    Returns a 2-tuple of:
        - unique duplicate files, also as 2-tuple of (file_id, count)
        - df without duplicate file rows
    """
    # Find all duplicate files
    dups = df[df.duplicated("file", keep=False)]["file"].tolist()
    # Create a unique list of duplicate file names in order of occurrence
    d = OrderedDict()
    for f in dups:
        if f not in d:
            d[f] = 0
        d[f] += 1
    return (list(d.items()), df.drop_duplicates(subset="file", keep=False))


def find_sfl_files(root):
    """Find all files with .sfl extension beneath root.

    Returns a list of sfl file paths relative to root.
    """
    root = os.path.expanduser(root)
    sfl_paths = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.endswith(".sfl"):
                sfl_paths.append(os.path.join(dirpath, f))
    return sorted(sfl_paths)


def fix(df):
    """Return a copy of df ready for db import.

    - Adds a "date" column if not present, based on filename datestamp
    - Adds or replaces day of year directory component of "file" values
    - Set any stream pressure values <= 0 to 1e-4 (small positive number)
    - Adds any missing db columns
    """
    newdf = df.copy(deep=True)

    # Add a date column if needed
    def date_from_file(f):
        try:
            d = seaflowfile.SeaFlowFile(f).rfc3339
        except sfperrors.FileError:
            d = ''
        return d

    if "date" not in newdf.columns:
        newdf["date"] = newdf["file"].map(date_from_file)

    # Add day of year directory if needed
    def dayofyear_from_file(f):
        try:
            d = seaflowfile.SeaFlowFile(f).file_id
        except sfperrors.FileError:
            d = f  # don't change anything if can't parse filename
        return d

    newdf["file"] = newdf["file"].map(dayofyear_from_file)

    # Convert stream pressure <= 0 to small positive number
    newdf.loc[newdf["stream_pressure"] <= 0, "stream_pressure"] = min_stream_pressure

    # Make sure all DB columns are present
    for k in colname_mapping["table_to_file"]:
        if k not in newdf.columns:
            newdf[k] = None

    return newdf


def fix_event_rate(df, event_counts):
    """
    Update event_rate field based on event counts in event_counts.

    Parameters
    -----------
    df: pandas DataFrame
        SFL DataFrame, based on a "fixed" file
    event_counts: dict of {str: int}
        Dictionary with file_id: event count

    Returns
    -------
    df: pandas DataFrame
        Copy of df with updated event_rate fields where possible.
    """
    newdf = df.copy(deep=True)
    for i, row in newdf.iterrows():
        try:
            file_duration = float(row["file_duration"])
        except ValueError:
            continue
        try:
            event_count = int(event_counts[row["file"]])
        except (ValueError, KeyError):
            continue
        try:
            newdf.loc[i, "event_rate"] = event_count / file_duration
        except ZeroDivisionError:
            newdf.iloc[i, "event_rate"] = 0.0
    return newdf


def has_gga(df):
    """Do any coordinates Series in this DataFrame contain GGA values?"""
    gga_lats = df["lat"].map(geo.is_gga_lat, na_action="ignore")
    gga_lons = df["lon"].map(geo.is_gga_lon, na_action="ignore")
    return (gga_lats | gga_lons).any()


def make_json_serializable(v):
    """Make sure v is JSON serializable if it's numpy type or plain object"""
    if isinstance(v, pd.np.generic):
        return pd.np.asscalar(v)
    return v


def parse_sfl_filename(fn):
    fn = os.path.basename(fn)
    m = re.match(r"^(?P<cruise>.+)_(?P<inst>[^_]+).sfl$", fn)
    if m:
        return (m.group('cruise'), m.group('inst'))
    return ()


@util.suppress_sigpipe
def print_json_errors(errors, fh, print_all=True):
    errors_output = []
    errors_seen = set()
    for e in errors:
        if (not print_all) and (e["message"] in errors_seen):
            continue
        errors_seen.add(e["message"])
        errors_output.append(e)
    fh.write(json.dumps(errors_output, sort_keys=True, indent=2, separators=(',', ':')))
    fh.write("\n")


@util.suppress_sigpipe
def print_tsv_errors(errors, fh, print_all=True):
    errors_output = []
    errors_seen = set()
    for e in errors:
        if (not print_all) and (e["message"] in errors_seen):
            continue
        errors_seen.add(e["message"])
        errors_output.append(e)

    # Find longest string in each column
    longest = {}
    for e in errors_output:
        for k, v in e.items():
            if k not in longest or longest[k] < len(k):
                longest[k] = len(k)
            if longest[k] < len(str(v)):
                longest[k] = len(str(v))

    # Write left-alignd space-spadded TSV output
    header_written = False
    for e in errors_output:
        if not header_written:
            texts = []
            for k in sorted(e.keys()):
                texts.append(k + (" " * (longest[k] - len(k))))
            print("\t".join(texts), file=fh)
            header_written = True
        texts = []
        for k in sorted(e.keys()):
            texts.append(str(e[k]) + (" " * (longest[k] - len(str(e[k])))))
        print("\t".join(texts), file=fh)


def read_file(file_path, convert_numerics=True, convert_colnames=True, **kwargs):
    """Parse SFL file into a DataFrame.

    Arguments:
    file -- SFL file path.

    Keyword arguments:
    convert_numerics -- Cast numeric SQL columns as numbers (default True).
    convert_colnames -- Remap file column names to match SFL SQL table column
        where appropriate. (default True).
    """
    defaults = {
        "sep": str(sfl_delim),
        "dtype": str,
        "na_filter": True,
        "encoding": "utf-8"
    }
    kwargs_defaults = dict(defaults, **kwargs)

    df = pd.read_csv(file_path, **kwargs_defaults)
    df = df.rename(columns=colname_mapping["file_to_table"])

    if convert_numerics:
        for colname in numeric_columns:
            df[colname] = pd.to_numeric(df[colname], errors='coerce')

    if not convert_colnames:
        # Revert column name mapping back to file convention
        df = df.rename(columns=colname_mapping["table_to_file"])

    return df


def save_to_db(df, dbpath, cruise=None, serial=None):
    """Write SFL dataframe to a SQLite3 database.

    Arguments:
    df -- SFL DataFrame.
    dbpath -- Path to SQLite3 database file.
    """
    db.create_db(dbpath)  # create or update db if needed
    if cruise is None:
        cruise = 'None'
    if serial is None:
        serial = 'None'
    metadf = pd.DataFrame({'cruise': [cruise], 'inst': [serial]})
    db.save_metadata(dbpath, metadf.to_dict('index').values())
    # This assumes there are column names which match SQL SFL table
    db.save_sfl(dbpath, df.to_dict('index').values())


@util.suppress_sigpipe
def save_to_file(df, outpath, convert_colnames=True, all_columns=False):
    """Write SFL dataframe to a csv file.

    Arguments:
    df -- SFL DataFrame.
    outpath -- Output file path.

    Keyword Arguments:
    convert_colnames -- Remap SQL table column names to SFL file column names
        where appropriate. (default True).
    """
    # Remove input file path and line number columns that may have been
    # added.
    if not all_columns:
        df = df[output_columns]
    if convert_colnames:
        df = df.rename(columns=colname_mapping["table_to_file"])
    df.to_csv(outpath, sep=str(sfl_delim), na_rep="NA", encoding="utf-8",
        index=False, float_format="%.4f")
