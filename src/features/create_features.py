import numpy as np


def __get_recur_day(day):
    return str((day - 1) % 7 + 1)


def __get_time_bin(hour):
    return "night" if hour < 5 or hour >= 22 \
        else "evening" if hour >= 18 \
        else "afternoon" if hour >= 10 \
        else "morning"


def create_features(df):
    df["recur_day"] = ((df.day - 1) % 7 + 1).astype(np.int8)

    df["time_bin"] = df.hour.map(__get_time_bin)

    df["recur_day_1"] = (df.recur_day == 1).astype(np.int8)
    df["recur_day_2"] = (df.recur_day == 2).astype(np.int8)
    df["recur_day_3"] = (df.recur_day == 3).astype(np.int8)
    df["recur_day_4"] = (df.recur_day == 4).astype(np.int8)
    df["recur_day_5"] = (df.recur_day == 5).astype(np.int8)
    df["recur_day_6"] = (df.recur_day == 6).astype(np.int8)
    df["recur_day_7"] = (df.recur_day == 7).astype(np.int8)

    df["time_bin_morning"] = (df.time_bin == "morning").astype(np.int8)
    df["time_bin_afternoon"] = (df.time_bin == "afternoon").astype(np.int8)
    df["time_bin_evening"] = (df.time_bin == "evening").astype(np.int8)
    df["time_bin_night"] = (df.time_bin == "night").astype(np.int8)

    return df
