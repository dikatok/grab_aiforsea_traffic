import pandas as pd


def impute_data(df, lat_long_map):
    max_day = df.day.max()
    max_day_hour = df[df.day == max_day].hour.max()
    max_day_minute = df[(df.day == max_day) & (df.hour == max_day_hour)].minute.max()

    all_time_points_df = pd.DataFrame({"day": [d for d in range(1, max_day + 1)]})\
        .assign(key=1)\
        .merge(pd.DataFrame({"hour": [h for h in range(0, 24)]})
               .assign(key=1)
               .merge(pd.DataFrame({"minute": [m for m in range(0, 60, 15)]}).assign(key=1),
                      on="key"),
               on="key")\
        .drop("key", axis=1)

    df = df.set_index(["geohash6", "lat", "long", "day", "hour", "minute"])\
        .reindex(pd.DataFrame([(geo, lat, long) for (geo, (lat, long)) in lat_long_map.items()],
                              columns=["geohash6", "lat", "long"])
                 .assign(key=1)
                 .merge(all_time_points_df.assign(key=1), on="key").drop("key", axis=1)
                 .set_index(["geohash6", "lat", "long", "day", "hour", "minute"])
                 .index)\
        .fillna(0)\
        .reset_index()

    df = df[(df.day < max_day)
            | ((df.day == max_day) & (df.hour < max_day_hour))
            | ((df.day == max_day) & (df.hour == max_day_hour) & (df.minute <= max_day_minute))]

    return df
