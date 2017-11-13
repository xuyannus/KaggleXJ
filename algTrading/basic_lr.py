import os
import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

TRAINING_DATA_PATH = os.path.dirname(__file__) + "/../data/training_with_features.csv"
TESTING_DATA_PATH = os.path.dirname(__file__) + "/../data/testing.csv"


class LinearRegressionModel:
    def __init__(self):
        self.x_cols_ask = []
        self.y_cols_ask = []

        self.x_cols_bid = []
        self.y_cols_bid = []

        self.y_pred_cols_bid = []
        self.y_pred_cols_ask = []

        self.train_df = None
        self.test_df = None

    def load_data(self):
        self.train_df = pd.read_csv(TRAINING_DATA_PATH)

        for i in range(52, 53):
            self.y_cols_bid.append("bid{}".format(i))
            self.y_cols_ask.append("ask{}".format(i))

        for i in range(52, 101):
            self.y_pred_cols_bid.append("bid{}".format(i))
            self.y_pred_cols_ask.append("ask{}".format(i))

        for i in range(1, 51):
            self.x_cols_bid.append("bid{}".format(i))
            self.x_cols_bid.append("ask{}".format(i))

            self.x_cols_ask.append("bid{}".format(i))
            self.x_cols_ask.append("ask{}".format(i))

        self.x_cols_bid.extend(['security_id', 'p_tcount', 'p_value', 'trade_vwap', 'trade_volume', 'initiator'])
        self.x_cols_ask.extend(['security_id', 'p_tcount', 'p_value', 'trade_vwap', 'trade_volume', 'initiator'])

        self.X_ask = self.train_df[self.x_cols_ask]
        self.y_ask = self.train_df[self.y_cols_ask]

        self.X_bid = self.train_df[self.x_cols_bid]
        self.y_bid = self.train_df[self.y_cols_bid]

    def data_transform(self):
        label_encoder = LabelEncoder()
        self.X_ask['initiator'] = label_encoder.fit_transform(self.X_ask['initiator'].values)
        self.X_bid['initiator'] = label_encoder.fit_transform(self.X_bid['initiator'].values)

        onehot_encoder = OneHotEncoder(categorical_features=[50, 55])
        self.X_ask = onehot_encoder.fit_transform(self.X_ask)
        self.X_bid = onehot_encoder.fit_transform(self.X_bid)

    def model_build(self):
        lr_model_bid = lm.LinearRegression()
        lr_model_bid.fit(self.X_bid.todense(), self.y_bid.values)
        estimated_y_bid = lr_model_bid.predict(self.X_bid.todense())

        bid_true_y = self.train_df[self.y_pred_cols_bid].values.flatten('F')
        bid_predicted_y = np.repeat(estimated_y_bid, 49)

        lr_model_ask = lm.LinearRegression()
        lr_model_ask.fit(self.X_ask.todense(), self.y_ask.values)
        estimated_y_ask = lr_model_bid.predict(self.X_ask.todense())

        ask_true_y = self.train_df[self.y_pred_cols_ask].values.flatten('F')
        ask_predicted_y = np.repeat(estimated_y_ask, 49)
        print("training RMSE: {}".format(sqrt(mean_squared_error(np.append(bid_true_y, ask_true_y), np.append(bid_predicted_y, ask_predicted_y)))))

    def model_prediction(self):
        pass


def linear_model_run():
    model = LinearRegressionModel()
    model.load_data()
    model.data_transform()
    model.model_build()
    model.model_prediction()


if __name__ == "__main__":
    linear_model_run()