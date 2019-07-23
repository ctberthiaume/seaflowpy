import click
import os
import time
import numpy as np
import seaflowpy as sfp
from seaflowpy import errors
from seaflowpy.fileio import file_open_r

def read_labview(path, columns, fileobj=None):
    colcnt = len(columns) + 2  # 2 leading column per row

    with file_open_r(path, fileobj) as fh:
        # Particle count (rows of data) is stored in an initial 32-bit
        # unsigned int
        try:
            buff = fh.read(4)
        except (IOError, EOFError) as e:
            raise errors.FileError("File could not be read: {}".format(str(e)))
        if len(buff) == 0:
            raise errors.FileError("File is empty")
        if len(buff) != 4:
            raise errors.FileError("File has invalid particle count header")
        rowcnt = np.frombuffer(buff, dtype="uint32", count=1)[0]
        if rowcnt == 0:
            raise errors.FileError("File has no particle data")

        # Read the rest of the data. Each particle has colcnt unsigned
        # 16-bit ints in a row.
        expected_bytes = rowcnt * colcnt * 2  # rowcnt * colcnt columns * 2 bytes
        # must cast to int here because while BufferedIOReader objects
        # returned from io.open(path, "rb") will accept a numpy.int64 type,
        # io.BytesIO objects will not accept this type and will only accept
        # vanilla int types. This is true for Python 3, not for Python 2.
        try:
            buff = fh.read(int(expected_bytes))
        except (IOError, EOFError) as e:
            raise errors.FileError("File could not be read: {}".format(str(e)))

        # Read any extra data at the end of the file for error checking. There
        # shouldn't be any extra data, btw.
        extra_bytes = 0
        while True:
            try:
                new_bytes = len(fh.read(8192))
            except (IOError, EOFError) as e:
                raise errors.FileError("File could not be read: {}".format(str(e)))
            extra_bytes += new_bytes
            if new_bytes == 0:  # end of file
                break

    # Check that file has the expected number of data bytes.
    found_bytes = len(buff) + extra_bytes
    if found_bytes != expected_bytes:
        raise errors.FileError(
            "File has incorrect number of data bytes. Expected %i, saw %i" %
            (expected_bytes, found_bytes)
        )

    events = np.frombuffer(buff, dtype="uint16", count=rowcnt*colcnt)
    # Reshape into a matrix of colcnt columns and one row per particle
    # The first two uint16s [0,10] from start of each row are left out.
    # These ints are an idiosyncrasy of LabVIEW's binary output format.
    # I believe they're supposed to serve as EOL signals (NULL,
    # linefeed in ASCII), but because the last line doesn't have them
    # it's easier to treat them as leading ints on each line after the
    # header.
    events = np.reshape(events, [rowcnt, colcnt])
    events = np.delete(events, [0, 1], 1)
    return events



def read_evt_labview(path, fileobj=None):
     return read_labview(path, sfp.particleops.COLUMNS, fileobj).astype(np.float64)



filter_params = {'id': {0: '87ff0de6-88da-4d44-ab87-9feb955a72d6', 1: '87ff0de6-88da-4d44-ab87-9feb955a72d6', 2: '87ff0de6-88da-4d44-ab87-9feb955a72d6'}, 'date': {0: '2019-06-20T18:10:22+00:00', 1: '2019-06-20T18:10:22+00:00', 2: '2019-06-20T18:10:22+00:00'}, 'quantile': {0: 2.5, 1: 50.0, 2: 97.5}, 'beads_fsc_small': {0: 52368.0, 1: 53056.0, 2: 53552.0}, 'beads_D1': {0: 29296.0, 1: 28048.0, 2: 27024.0}, 'beads_D2': {0: 32608.0, 1: 29744.0, 2: 27260.0}, 'width': {0: 5000.0, 1: 5000.0, 2: 5000.0}, 'notch_small_D1': {0: 0.5589999999999999, 1: 0.529, 2: 0.505}, 'notch_small_D2': {0: 0.623, 1: 0.561, 2: 0.509}, 'notch_large_D1': {0: 1.64, 1: 1.635, 2: 1.63}, 'notch_large_D2': {0: 1.6369999999999998, 1: 1.632, 2: 1.6269999999999998}, 'offset_small_D1': {0: 0.0, 1: 0.0, 2: 0.0}, 'offset_small_D2': {0: 0.0, 1: 0.0, 2: 0.0}, 'offset_large_D1': {0: -56588.0, 1: -58699.0, 2: -60266.0}, 'offset_large_D2': {0: -53118.0, 1: -56843.0, 2: -59869.0}}

@click.command()
@click.option("-v", "--verbose", is_flag=True,
    help="Print each file examined.")
@click.argument("files", nargs=-1, type=click.Path())
def cmd(verbose, files):
    t0 = time.time()
    for f in files:
        try:
            sfile = sfp.seaflowfile.SeaFlowFile(f)
        except sfp.errors.FileError:
            continue
        if sfile.is_evt:
            try:
                events = sfp.fileio.read_evt_labview(f)
                msg = f"{os.path.basename(f)} {events.shape[0]}"
                if verbose:
                    print(msg)
            except sfp.errors.FileError:
                pass
    print("{}".format(time.time() - t0))

if __name__ == "__main__":
    cmd()
