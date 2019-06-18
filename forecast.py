import argparse
import pandas as pd
import tensorflow as tf

from src.data.prepare import prepare_data
from src.features.create_features import __get_time_bin


def join_pred(series, pred):
    sample = series[0, -1, :].copy()

    # increment timestamp
    sample[2] = sample[2] + 1 if sample[3] == 23 and sample[4] == 45 else sample[2]
    sample[3] = 0 if sample[3] == 23 and sample[4] == 45 else sample[3] if sample[4] < 45 else sample[3] + 1
    sample[4] = (sample[4] + 15) % 60

    # demand
    sample[5] = pred

    # recur day
    sample[6:7] = [1 if recur == ((sample[2] - 1) % 7) + 1 else 0
                   for recur in range(1, 8)]

    # time bin
    sample[13:4] = [1 if time_bin == __get_time_bin(sample[3]) else 0
                    for time_bin in ["morning", "afternoon", "evening", "night"]]

    series.append(sample)

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

    model = tf.keras.models.load_model("./models/final.h5")

    test_df = pd.read_csv(csv_path)

    test_df = prepare_data(test_df)

    test_geohashes = test_df.geohash6.unique()

    result = []

    for geo in test_geohashes:
        cur_geo = test_df[test_df.geohash6 == geo]
        geo_series = []
        preds = []
        for i in range(0, 5):
            pred = model.predict(geo_series)
            geo_series = join_pred(geo_series, pred)
            preds.append(pred)
        result.append(preds)

    pd.DataFrame(result).to_csv(out_path)
