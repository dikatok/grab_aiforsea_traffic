import argparse
import pandas as pd
import tensorflow as tf
import numpy as np

from src.data.prepare import prepare_data
from src.features.create_features import __get_time_bin
from src.models.model import Model
from src.features.create_features import create_features


def join_pred(series, pred):
    sample = series[0, -1, :].copy()

    # increment timestamp
    sample[2] = sample[2] + 1 if sample[3] == 23 and sample[4] == 45 else sample[2]
    sample[3] = 0 if sample[3] == 23 and sample[4] == 45 else sample[3] if sample[4] < 45 else sample[3] + 1
    sample[4] = (sample[4] + 15) % 60

    # demand
    sample[5] = pred

    # recur day
    sample[6:13] = [1 if recur == ((sample[2] - 1) % 7) + 1 else 0
                    for recur in range(1, 8)]

    # time bin
    sample[13:17] = [1 if time_bin == __get_time_bin(sample[3]) else 0
                     for time_bin in ["morning", "afternoon", "evening", "night"]]

    series = np.append(series, np.reshape(sample, [1, 1, -1]), axis=1)

    return series


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='Path to model directory', default="models")
    parser.add_argument('--csv', help='Path to csv file to be forecast', required=True)
    parser.add_argument('--out', help='Path to output csv file', required=True)
    args = parser.parse_args()

    models_path = args.model
    csv_path = args.csv
    out_path = args.out

    model = Model()
    model.checkpoint.restore(tf.train.latest_checkpoint(models_path)).expect_partial()

    test_df = pd.read_csv(csv_path).iloc[:100000]
    test_df = prepare_data(test_df, impute=True)
    test_df = test_df[["geohash6", "lat", "long", "day", "hour", "minute", "demand"]]
    test_df = create_features(test_df)
    test_df = test_df.sort_values(["day", "hour", "minute"])

    feature_columns = test_df.columns.drop(["geohash6", "recur_day", "time_bin"])
    test_geohashes = test_df.geohash6.unique()
    result_df = pd.DataFrame(columns=["geohash6", "day", "timestamp", "demand"])

    for geo in test_geohashes:
        geo_series = np.expand_dims(np.array(test_df[test_df.geohash6 == geo][feature_columns]), axis=0)

        for i in range(0, 5):
            pred, _ = model(geo_series)
            geo_series = join_pred(geo_series, np.squeeze(pred))
            last_result = geo_series[0, -1, [2, 3, 4, 5]]
            result_df = result_df.append(dict(geohash6=geo, demand=last_result[3],
                                              day=int(last_result[0]),
                                              timestamp=f"{int(last_result[1])}:{int(last_result[2])}"),
                                         ignore_index=True)
        break

    result_df.to_csv(out_path, index=False)
