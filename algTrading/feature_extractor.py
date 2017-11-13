import pandas as pd

TRAINING_DATA_PATH = "../data/training.csv"
TRAINING_DATA_SAVE_TO_PATH = "../data/training_with_features.csv"


class FeatureExtractor:
    def __init__(self):
        self.train_df = None

    def load_data(self):
        self.train_df = pd.read_csv(TRAINING_DATA_PATH)

    def extract_features(self):
        # mid price & spread
        for i in range(1, 101):
            ask_col = 'ask{}'.format(i)
            bid_col = 'bid{}'.format(i)

            mid_col = 'mid{}'.format(i)
            spd_col = 'spd{}'.format(i)

            self.train_df[mid_col] = (self.train_df[ask_col] + self.train_df[bid_col]) / 2.0
            self.train_df[spd_col] = (self.train_df[ask_col] - self.train_df[bid_col])

        spd_columns_till_49 = []
        for i in range(1, 50):
            spd_columns_till_49.append("spd{}".format(i))

        self.train_df['max_spd'] = self.train_df[spd_columns_till_49].max(axis=1)

    def save_data(self):
        self.train_df.to_csv(TRAINING_DATA_SAVE_TO_PATH, encoding='utf-8', index=False)


def feature_extract():
    tool = FeatureExtractor()
    tool.load_data()
    tool.extract_features()
    tool.save_data()


if __name__ == "__main__":
    feature_extract()
