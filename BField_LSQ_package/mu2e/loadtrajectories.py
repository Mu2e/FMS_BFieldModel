#! /usr/bin/env python
"""Module for loading in a DataFrame of trajectory data points from datafiles created
using mu2e/tools/particletransport.py

This module defines functions that takes trajectory-style input files (.pkl),
and generates :class:`pandas.DataFrame` for use in this Python Mu2E package. In particular
these DataFrames are used in analyzing signal electron tracks.
Some runs have multiple iterations of the same event (e.g. run 04 iterates on Bfield shifts),
which motivates the need for this tool.
The input and output data are saved to a subdir of the user-defined `mu2e_ext_path`, and are not committed to github.

Example:

Todo:
    * Generate example.


*2019 Cole Kampa, Northwestern University*

colekampa2024@u.northwestern.edu
"""


import re
import os
import pandas as pd

from mu2e import mu2e_ext_path

def LoadTrajectory(run=5, subrun=None, event=0, form='sparse', stride=1):
    good_runs = [4,5]
    if run not in good_runs:
        raise Exception(f"Please enter a valid run number in {good_runs}.")
    if subrun == None:
        data_dir = mu2e_ext_path+f"trajectory/run{run:02d}/{form}/"
    else:
        data_dir = mu2e_ext_path+f"trajectory/run{run:02d}/{form}/subrun{subrun:02d}/"

    files_finder = re.compile(f".*_{event:03d}.*") # with the simple filename format as of 7/29/19 this regex expression will find all relevant files for a given event
    event_files = sorted([data_dir+f for f in os.listdir(data_dir) if files_finder.match(f)])

    df = [] # first load all DataFrames into a list called df, then concatenate
    for file in event_files:
        df.append(pd.read_pickle(file).iloc[::stride,:])
    if len(df) > 1:
        df = pd.concat(df, join='outer', ignore_index=True, sort = False) # concatenate!
    elif len(df) == 0:
        raise Exception("No event file was found.")
    else:
        df = df[0]

    return df
