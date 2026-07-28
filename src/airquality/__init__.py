"""Airquality package."""

import logging


logging.getLogger("prophet.plot").addFilter(
    lambda record: record.getMessage()
    != "Importing plotly failed. Interactive plots will not work."
)
