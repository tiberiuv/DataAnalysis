import keras.layers as kl
from keras.models import Model
from keras import regularizers
import pandas as pd
import numpy as np
# np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from bokeh.plotting import output_file, figure, show


class NeuralNetwork:
    def __init__(self, input_shape, stock_or_return):
        self.input_shape = input_shape
        self.stock_or_return = stock_or_return

    def make_train_model(self, epochs=1):
        input_data = kl.Input(shape=(1, self.input_shape))
        lstm = kl.LSTM(5, input_shape=(1, self.input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.003),
                       recurrent_regularizer=regularizers.l2(0), dropout=0.2, recurrent_dropout=0.2)(input_data)
        perc = kl.Dense(5, activation="sigmoid",
                        activity_regularizer=regularizers.l2(0.005))(lstm)
        lstm2 = kl.LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
                        dropout=0.2, recurrent_dropout=0.2)(perc)
        # perc2 = kl.Dense(2, activation="sigmoid", activity_regularizer=regularizers.l2(0.005))(lstm2)
        # lstm3 = kl.LSTM(2, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
        #                 dropout=0.2, recurrent_dropout=0.2)(perc2)
        out = kl.Dense(1, activation="sigmoid",
                       activity_regularizer=regularizers.l2(0.001))(lstm2)

        model = Model(input_data, out)
        model.compile(optimizer="adam",
                      loss="mean_squared_error", metrics=["mse"])

        # load data

        train = np.reshape(np.array(pd.read_csv("features/autoencoded_train_data.csv", index_col=0)),
                           (len(np.array(pd.read_csv("features/autoencoded_train_data.csv"))), 1, self.input_shape))
        train_y = np.array(pd.read_csv(
            "features/autoencoded_train_y.csv", index_col=0))

        # train model
        model.fit(train, train_y, epochs=epochs)

        model.save("models/model.h5", overwrite=True, include_optimizer=True)

        test_x = np.reshape(np.array(pd.read_csv("features/autoencoded_test_data.csv", index_col=0)),
                            (len(np.array(pd.read_csv("features/autoencoded_test_data.csv"))), 1, self.input_shape))
        test_y = np.array(pd.read_csv(
            "features/autoencoded_test_y.csv", index_col=0))

        stock_data_test = np.array(pd.read_csv(
            "stock_data_test.csv", index_col=0))

        print(model.evaluate(test_x, test_y))
        prediction_data = []
        stock_data = []
        for i in range(len(test_y)):
            prediction = (model.predict(np.reshape(
                test_x[i], (1, 1, self.input_shape))))
            prediction_data.append(np.reshape(prediction, (1,)))
            prediction_corrected = (
                prediction_data - np.mean(prediction_data)) / np.std(prediction_data)
            stock_price = np.exp(np.reshape(
                prediction, (1,))) * stock_data_test[i]
            stock_data.append(stock_price[0])
        stock_data[:] = [
            i - (float(stock_data[0]) - float(stock_data_test[0])) for i in stock_data]

        if self.stock_or_return:
            actual_data = pd.DataFrame(stock_data_test)
            # actual_data[0].plot(label='Actual', figsize=(
            #     16, 8), title='Prediction vs Actual')

            predicted_data = pd.DataFrame(stock_data)
            # print(predicted_data[0].head(4))

            stock = pd.DataFrame(stock_data, index=None)
            stock.to_csv("sample_predictions/AAPL_predicted_prices.csv")
            stock_test = pd.DataFrame(stock_data_test, index=None)
            stock_test.to_csv("sample_predictions/AAPL_actual_prices.csv")

            # For gettting date
            adj_close = actual_data.ix[0, 0]
            print('adj close value' + str(adj_close))

            df = pd.read_csv("stock_data.csv")
            row = df.loc[df['Adj Close'] == adj_close]
            print(row)
            a = row['Date'].to_string()
            b = a.split('    ', 1)
            print(b[1])
            df_date = df.loc[(df['Date'] >= b[1])]
            c = df_date['Date']
            print(c.head(4))
            d = []
            for index, row in df_date.iterrows():
                d.append(row['Date'])

            act_stock = []
            for index, row in actual_data.iterrows():
                act_stock.append(row[0])

            my_dict = {'Date': d,
                       'stock': act_stock}

            actual_data_date = pd.DataFrame(my_dict)

            p_stock = []

            for index, row in predicted_data.iterrows():
                p_stock.append(row[0])
            d = d[:-1]
            my_dict1 = {'Date': d,
                        'stock': p_stock}

            #actual_data_date = actual_data_date.join(actual_data[0])
            actual_data_date.to_csv('a.csv', index=False)

            a = (pd.read_csv('a.csv', index_col=False))
            predicted_data_date = pd.DataFrame(my_dict1)

            # actual_data_date.to_csv("sample_predictions/a.csv")
            # df = pd.read_csv("sample_predictions/a.csv")

            # print(df)
            a['stock'].plot(label='Actual', figsize=(
                16, 8), title='Prediction vs Actual')
            predicted_data_date['stock'].plot(label='Predicted Price')
            # print(actual_data_date)

            plt.legend()
            plt.show()
        else:
            plt.plot(prediction_data)
            plt.plot(test_y)
            plt.show()


if __name__ == "__main__":
    model = NeuralNetwork(20, True)
    model.make_train_model()
