# AI for SEA (Traffic Management)

[Status] Still in progress, got sudden Cuda related problems

## How to Test
Create new conda environment and activate it

`conda env create -f environment.yml`

`conda activate test`

Download the model checkpoint

`git lfs pull`

Perform forecasting

`python forecast.py --csv test.csv`

## References
https://www.kaggle.com/c/web-traffic-time-series-forecasting/discussion/43795#latest-528073

https://www.kaggle.com/humamfauzi/multiple-stock-prediction-using-single-nn

https://eng.uber.com/neural-networks/

https://machinelearningmastery.com/lstm-model-architecture-for-rare-event-time-series-forecasting/