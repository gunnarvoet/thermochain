r"""In-situ sensor calibration and general data processing for high-density moored thermistor strings.

# Overview

This software package applies sensor calibration and general depth-gridding to a set of data files collected with moored thermistors.
The in-situ sensor calibration correcting for sensor offset and sensor drift with time at the core of this package follows the method developed in [Cimatoribus et. al. (20116)](https://journals.ametsoc.org/view/journals/atot/33/7/jtech-d-15-0243_1.xml).
For thermistor strings with sufficient sensor density (depending on ambient stratification between less than one to several meters) the in-situ calibration method allows for measuring temperature variability at $\mathcal{O}(10^{-4})$ K.

Currently the package works with raw data files from RBR Solo and SBE 56 thermistors.

Processing steps are
- conversion
- clock calibration
- CTD rosette calibration
- depth and time gridding
- sensor drift calibration

Individual steps are detailed in the following.

# Raw file conversion

Initialize a processing object to process data from one mooring with `thermodrift.io.ProcessThermistorMooring` using a `.yaml`-configuration file. The processing object will:

- Read the  configuration file (one per mooring) that contains paths to data (both raw input & output) directories, sensor and mooring spreadsheets, and other processing parameters.
- Read sensor information from the sensor spreadsheet. See [templates](#templates) in both `.csv` and `.xlsx` format in the `templates` folder. Either of them works, just make sure to keep the column names as they are. Also, keeping any datetime columns in plain text format and sticking to `yyyy-mm-dd hh:mm:ss` format will help with parsing the spreadsheet. The sensor spreadsheet may contain sensors from several moorings but only one row per serial number.

# Clock calibration

# CTD rosette calibration

# Depth and time gridding

# Sensor drift calibration

# Templates

The `templates` directory holds templates for
- Sensor spreadsheet (holding all per-sensor info, e.g. clock calibration times, CTD calibration times, specific pre- and post-deployment notes.
- Mooring configuration spreadsheet, laying out the configuration of the sensors on the moorings.

# License
.. include:: ../../LICENSE
"""

__author__ = """Gunnar Voet"""
__email__ = "gvoet@ucsd.edu"
__version__ = "2023.12.0"

__all__ = ["io", "plot"]
from . import io, plot
