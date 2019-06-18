import numpy as np


def create_dataset(df, epochs, batch_size=32, look_back=14):
    batch_seq = df.geohash6.unique()
    num_batch_seq = len(batch_seq)
    max_day = df.day.max()
    feature_columns = df.columns.drop(["geohash6", "recur_day", "time_bin"])

    for d in range(1, max_day + 1):
        for h in range(0, 23):
            for m in range(0, 60, 15):
                if d == max_day and h == 23 and m == 45:
                    break

                next_timestamp = (d + 1, 0, 0) if h == 23 and m == 45 else (d, h + 1 if m == 45 else h, 0 if m == 45 else m + 15)

                for _ in range(epochs):
                    inputs_df = df[(df.day <= d) & (df.day >= d - look_back) & (df.hour <= h) & (df.minute <= m)]
                    outputs_df = df[(df.day == next_timestamp[0])
                                    & (df.hour == next_timestamp[1])
                                    & (df.minute == next_timestamp[2])]

                    np.random.shuffle(batch_seq)
                    for b in range(0, num_batch_seq, batch_size):
                        if b == 0:
                            batch_geos = batch_seq[:batch_size]
                        else:
                            batch_geos = batch_seq[b - batch_size:b] if b <= num_batch_seq else batch_seq[
                                                                                                num_batch_seq - batch_size:]

                        inputs = []
                        outputs = []
                        for geo in batch_geos:
                            inputs.append(np.array(inputs_df[inputs_df.geohash6 == geo][feature_columns]))
                            outputs.append(np.array(outputs_df[outputs_df.geohash6 == geo].demand))

                        yield np.array(inputs), np.array(outputs)
    #
    # for d in range(look_back, max_day, 1):
    #     for _ in range(epochs):
    #         h, m = np.random.choice(list(range(0, 24, 1))), np.random.choice(list(range(0, 60, 15)))
    #         h = h - 1 if d == max_day else h
    #
    #         next_timestamp = (d + 1, 0, 0) if h == 23 and m == 45 else (d, h + 1 if m == 45 else h, 0 if m == 45 else m + 15)
    #
    #         inputs_df = df[(df.day <= d) & (df.day >= d - look_back) & (df.hour <= h) & (df.minute <= m)]
    #         outputs_df = df[(df.day == next_timestamp[0])
    #                         & (df.hour == next_timestamp[1])
    #                         & (df.minute == next_timestamp[2])]
    #
    #         np.random.shuffle(batch_seq)
    #         for b in range(0, num_batch_seq, batch_size):
    #             if b == 0:
    #                 batch_geos = batch_seq[:batch_size]
    #             else:
    #                 batch_geos = batch_seq[b - batch_size:b] if b <= num_batch_seq else batch_seq[num_batch_seq - batch_size:]
    #
    #             inputs = []
    #             outputs = []
    #             for geo in batch_geos:
    #                 inputs.append(np.array(inputs_df[inputs_df.geohash6 == geo][feature_columns]))
    #                 outputs.append(np.array(outputs_df[outputs_df.geohash6 == geo].demand))
    #
    #             yield np.array(inputs), np.array(outputs)
