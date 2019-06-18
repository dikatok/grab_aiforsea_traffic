# AI for SEA (Traffic Management)

##How To

### Environment

`conda env create -f environment.yml`

`conda activate test`

### View logs

`tensorboard --logdir logs`

### Perform forecasting

`python forecast.py --csv test.csv --out out.csv`

## Future recommendations
1. Use sigmoid activation on the output
2. Use validation set, eg: 1 day after every train set
3. Improve data generator to remove data to model bottleneck
4. Use independent autoencoder + forecaster model
5. Use cross features (eg: recur_day x time_bin)
6. Use feature embeddings

## References
https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795#latest-528073

https://www.kaggle.com/humamfauzi/multiple-stock-prediction-using-single-nn

https://eng.uber.com/neural-networks/

https://machinelearningmastery.com/lstm-model-architecture-for-rare-event-time-series-forecasting/