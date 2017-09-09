import numpy as np
import pandas as pd


class PipeLine(object):
    """This class deals with the row data and does all feature extraction and
    preprocessing."""
    cols = ['date', 'time', 'open', 'high', 'low', 'close', 'volume']
    smas = [3, 5, 10]

    def __init__(self, data, header='infer', index_col=None,
                 names=cols, price_df=None, final=None):
        """
        Parameters
        ----------
        data : string containg the csv file containing the price data
        header : int or list of ints, default 'infer'
        names : array-like, default None List of column names to use.
        index_col : int or sequence or False, default None
        price_df : DataFrame in a tidy form conaining datatime index and column names
        final: DataFrame containing the price data and the newly genrated features data
        """
        self.data = data
        self.header = header
        self.names = names
        self.index_col = index_col
        self.price_df = price_df
        self.final = final
        self.y = None

    def read_data(self):
        """Read_data and name the columns."""
        self.price_df = pd.read_csv(self.data, header=self.header,
                                    index_col=self.index_col,
                                    names=self.names)

    def set_datetime_index(self):
        """Creat a date time index and drop the date and time coloumns."""
        dt = self.price_df.date.str.replace('.', '-')\
            + ' ' + \
            self.price_df.time

        self.price_df.set_index(dt, inplace=True)
        self.price_df.drop(['date', 'time'], axis=1, inplace=True)

    def make_features(self, smas=smas, pred_mode=False):
        """Append all features to a dataframe."""
        data = self.price_df.copy()
        data['U_D'] = PipeLine.up_or_down(data)
        data['up_shadow'] = PipeLine.up_shadow(data)
        data['lo_shadow'] = PipeLine.lo_shadow(data)
        data['body'] = PipeLine.body(data)
        data['range'] = PipeLine.candle_range(data)
        data['price'] = PipeLine.median_price(data)
        data['price_change'] = PipeLine.price_change(data)
        data = pd.concat([data, PipeLine.make_sma(data)[0]], axis=1)
        data = pd.concat([data, PipeLine.sma_change(data)], axis=1)

        # drop the sma_n columns for avoiding data leakage.
        data.drop(PipeLine.make_sma(data)[1], axis=1, inplace=True)

        if not pred_mode:
            data['y'] = data.U_D.shift(-1)
            data.dropna(inplace=True)
            self.y = data.pop('y')
            data.drop(['price', 'high', 'low', 'open', 'close'], axis=1, inplace=True)
            self.final = data
        else:
            data.dropna(inplace=True)
            data.drop(['price', 'high', 'low', 'open', 'close'], axis=1, inplace=True)
            self.final = data

    @staticmethod
    def up_or_down(data):
        """Up or down."""
        return np.where(data['open'] > data['close'], 1, 0)

    @staticmethod
    def up_shadow(data):
        """Return the upper shadow."""
        return -np.where(data['U_D'] == 1, data.high - data.close, data.high - data.open)

    @staticmethod
    def lo_shadow(data):
        """Return the lower shadow."""
        return np.where(data['U_D'] == 1, data.open - data.low, data.close - data.low)

    @staticmethod
    def body(data):
        """Return the body of the candle."""
        return np.where(data['U_D'] == 1, data.close - data.open, data.close - data.open)

    @staticmethod
    def candle_range(data):
        """Return the range of the candle."""
        return data.high - data.low

    @staticmethod
    def median_price(data):
        """Median of [open,close,high,low]."""
        return data[['open', 'close', 'high', 'low']].median(axis=1)

    @staticmethod
    def price_change(data):
        """Return the median price of the currnt candle and the median price of the last candle subtracted from it."""
        previous_candle = data.price.shift(1)
        return data.price - previous_candle

    @staticmethod
    def make_sma(data, SMAs=smas):
        """Take a list of simple moving averages you want to be created."""
        if not isinstance(SMAs, list):
            print('SMAs has to be of type "list"')

        def_df = pd.DataFrame()

        # create sma for all values in SMAs
        for sma in SMAs:
            def_df['sma_{}'.format(sma)] = data.price.rolling(sma).mean()

        # Return coloumns that starts with sma
        return def_df[[col for col in def_df.columns if 'sma_' in col]], def_df.columns

    @staticmethod
    def sma_change(data):
        """Calculate the difference between the currnt value of the SMA and the last one."""
        sma_cols = [col for col in data.columns if 'sma_' in col]
        shifted = data[sma_cols].shift(1)
        sma_change_df = data[sma_cols] - shifted
        # renaming the coloumns
        sma_change_df.columns = [i + '_change' for i in sma_change_df.columns]
        return sma_change_df
