"""Functions manage SeaFlow sqlite3 DBs."""
import sqlite3 as sq


def ensure_opp_table(dbpath):
    """Ensure opp table exists."""
    con = sq.connect(dbpath)
    con.execute("""CREATE TABLE IF NOT EXISTS opp (
      -- First three columns are the EVT, OPP, VCT composite key
      cruise TEXT NOT NULL,
      file TEXT NOT NULL,  -- in old files, File+Day. in new files, Timestamp.
      particle INTEGER NOT NULL,
      -- Next we have the measurements. For these, see
      -- https://github.com/fribalet/flowPhyto/blob/master/R/Globals.R and look
      -- at version 3 of the evt header
      time INTEGER NOT NULL,
      pulse_width INTEGER NOT NULL,
      D1 REAL NOT NULL,
      D2 REAL NOT NULL,
      fsc_small REAL NOT NULL,
      fsc_perp REAL NOT NULL,
      fsc_big REAL NOT NULL,
      pe REAL NOT NULL,
      chl_small REAL NOT NULL,
      chl_big REAL NOT NULL,
      PRIMARY KEY (cruise, file, particle)
    )""")
    con.commit()
    con.close()

def ensure_opp_evt_ratio_table(dbpath):
    """Ensure opp_evt_ratio table exists."""
    con = sq.connect(dbpath)
    con.execute("""CREATE TABLE IF NOT EXISTS opp_evt_ratio (
      cruise TEXT NOT NULL,
      file TEXT NOT NULL,
      ratio REAL,
      PRIMARY KEY (cruise, file)
    )""")
    con.commit()
    con.close()
