import os
ROOT = r'%s' % os.path.abspath(os.path.join(
    os.path.dirname("src"))).replace('\\', '/')
DATA_PATH = r'%s' % os.path.abspath(os.path.join(
    os.path.dirname("src"), 'data')).replace('\\', '/')

# seed
SEED = 28
VERBOSE = 28

# data
DATA_FILE_STATION = "scr/data/_station_data.csv"
DATA_FILE_TRIP = "scr/data/trip_data.csv"
DATA_FILE_WEATHER = "data/data/weather_data.csv"

CLEAN_DATA = "/clean_data/cleandata.csv"
MODEL_NAME = "/model/rf_model"
