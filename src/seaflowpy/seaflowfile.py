from __future__ import absolute_import
from builtins import object
import arrow
import gzip
import io
import json
import os
import pprint
import re
from . import errors
from . import util
from collections import OrderedDict
from operator import itemgetter


julian_re = r'^\d{1,4}_\d{1,3}$'
new_path_re = r'^\d{1,4}_\d{1,3}/\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}[+-]\d{2}-\d{2}$'
new_file_re = r'^(?P<date>\d{4}-\d{2}-\d{2})T(?P<hours>\d{2})-(?P<minutes>\d{2})-(?P<seconds>\d{2})(?P<tzhours>[+-]\d{2})-(?P<tzminutes>\d{2})$'
old_path_re = r'^\d{1,4}_\d{1,3}/\d+\.evt$'
old_file_re = r'^\d+\.evt$'


class SeaFlowFile(object):
    """Base class for EVT/OPP/VCT file classes"""

    def __init__(self, path=None, fileobj=None):
        # If fileobj is set, read data from this object. The path will be used
        # to set the file name in the database and detect compression.
        self.path = path  # file path, local or in S3
        self.fileobj = fileobj  # data in file object

        if self.path:
            parts = parse_path(self.path)
            self.filename = parts["file"]
            self.filename_noext = remove_ext(parts["file"])

            if not (self.is_old_style or self.is_new_style):
                raise errors.FileError("Filename doesn't look like a SeaFlow file")

            if self.is_new_style:
                try:
                    self.date = date_from_filename(self.filename_noext)
                except ValueError as e:
                    raise errors.FileError("Error parsing date from filename: {}".format(e))
            else:
                self.date = None

            # YYYY_juliandayofyear directory found in file path and parsed
            # from file datestmap
            self.path_julian = parts["julian"]
            self.julian = create_julian_directory(self.date)

            # Identifer to match across file types (EVT/OPP/VCT)
            # Should be something like 2014_142/42.evt for old files. Note always
            # .evt even for opp and vct. No .gz.
            # Should be something like 2014_342/2014-12-08T22-53-34+00-00 for new
            # files. Note no extension including .gz.
            # The julian day directory will be based on parsed datestamp in
            # filename when possible, not the given path. The file ID based on
            # the given path is stored in path_file_id.
            if self.is_old_style:
                # path_file_id and file_id are always the same for old-style
                # filenames since we can't parse dates to calculate a julian
                # directory
                if self.path_julian:
                    self.file_id = "{}/{}".format(self.path_julian, self.filename_noext)
                else:
                    self.file_id = self.filename_noext
                self.path_file_id = self.file_id
            else:
                self.file_id = "{}/{}".format(self.julian, self.filename_noext)
                if self.path_julian:
                    self.path_file_id = "{}/{}".format(self.path_julian, self.filename_noext)
                else:
                    self.path_file_id = self.filename_noext


    def __str__(self):
        keys = ["path", "file_id"]
        return "SeaFlowFile: {}, {}".format(self.file_id, self.path)

    @property
    def isgz(self):
        """Is file gzipped?"""
        return self.path and self.filename.endswith(".gz")

    @property
    def is_old_style(self):
        """Is this old style file? e.g. 2014_185/1.evt."""
        return bool(re.match(old_file_re, self.filename_noext))

    @property
    def is_new_style(self):
        """Is this a new style file? e.g. 2018_082/2018-03-23T00-00-00+00-00.evt.gz"""
        return bool(re.match(new_file_re, self.filename_noext))

    def open(self):
        """Return a file-like object for reading."""
        handle = None
        if self.fileobj:
            if self.isgz:
                handle = gzip.GzipFile(fileobj=self.fileobj)
            else:
                handle = self.fileobj
        else:
            if self.isgz:
                handle = gzip.GzipFile(self.path)
            else:
                handle = io.open(self.path, 'rb')
        return handle

    @property
    def rfc3339(self):
        """Return RFC 3339 YYYY-MM-DDThh:mm:ss[+-]hh:mm parsed from filename"""
        if self.date:
            return arrow.Arrow.fromdatetime(self.date).format("YYYY-MM-DDTHH:mm:ssZZ")

    @property
    def sort_key(self):
        # Julian from filename if possible first, then from path, then nothing
        if self.julian:
            year, day = [int(x) for x in self.julian.split("_")]
        elif self.path_julian:
            year, day = [int(x) for x in self.path_julian.split("_")]
        else:
            year, day = 0, 0
        if self.is_old_style:
            # Number part of basename, necessary because number isn't
            # zero-filled
            file_key = int(self.filename_noext.split(".")[0])
        else:
            file_key = self.filename_noext
        return (year, day, file_key)



def create_julian_directory(dt):
    """Create SeaFlow julian dated directory from a datetime object"""
    if dt:
        return "{}_{}".format(dt.year, dt.strftime('%j'))


def date_from_filename(filename):
    """Return a datetime object based on new-style SeaFlow filename.

    Parts of the filename after the datestamp will be ignored.
    """
    filename_noext = remove_ext(os.path.basename(filename))
    m = re.match(new_file_re, filename_noext)
    if m:
        # New style EVT filenames, e.g.
        # - 2014-05-15T17-07-08+00-00
        # - 2014-05-15T17-07-08-07-00
        # Parse RFC 3339 date string with arrow, then get datetime
        date  = arrow.get("{}T{}:{}:{}{}{}".format(*m.groups())).datetime
        return date
    raise ValueError('filename does not look like a new-style SeaFlow file')


def parse_path(file_path):
    """Return a dict with entries for 'julian' dir and 'file' name"""
    d = { "julian": None, "file": None }
    parts = util.splitpath(file_path)
    d["file"] = parts[-1]
    if len(parts) > 1:
        if re.match(julian_re, parts[-2]):
            d["julian"] = parts[-2]
    return d


def remove_ext(filename):
    """Remove extensions from filename except .evt in old files."""
    file_parts = filename.split(".")
    noext = file_parts[0]
    if len(file_parts) > 1 and file_parts[1] == "evt" and re.match(r'^\d+$', file_parts[0]):
        # For old-style evt filenames, e.g. 42.evt
        noext += ".evt"
    return noext


def sorted_files(files):
    """Sort EVT/OPP/VCT file paths in chronological order.

    Order is based on julian day directory parsed from path and then file name.
    """
    sfiles = [SeaFlowFile(f) for f in files]
    return [s.path for s in sorted(sfiles, key=lambda x: x.sort_key)]