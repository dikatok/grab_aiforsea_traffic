def decode_geohash(df, lat_long_map):
    df["lat"] = df.geohash6.apply(lambda g: lat_long_map[g][0])
    df["long"] = df.geohash6.apply(lambda g: lat_long_map[g][1])

    return df


def decode_timestamp(df):
    df["hour"] = df.timestamp.map(lambda t: t.split(":")[0])
    df["minute"] = df.timestamp.map(lambda t: t.split(":")[1])

    return df
