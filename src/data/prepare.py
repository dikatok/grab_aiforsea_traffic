import numpy as np

from geohash2 import decode_exactly
from .decode import decode_geohash, decode_timestamp
from .impute import impute_data


def prepare_data(df):
    lat_long_map = {g: decode_exactly(g)[:2] for g in df.geohash6.unique()}

    df = decode_geohash(df, lat_long_map)

    df = decode_timestamp(df)

    df = df.astype({"day": np.int8, "hour": np.int8, "minute": np.int8})

    df = df.drop("timestamp", axis=1)

    df = impute_data(df, lat_long_map)

    return df
