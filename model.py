import keras.layers as kl
from keras.models import Model
from keras import regularizers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


class NeuralNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def make_train_model(self, epochs=1000):
        input_data = kl.Input(shape=(1, self.input_shape))
        lstm = kl.LSTM(
            16, input_shape=(1, self.input_shape), return_sequences=True, activity_regularizer=regularizers.l2(0.01),
            recurrent_regularizer=regularizers.l2(0.001), dropout=0.2, recurrent_dropout=0.2
        )(input_data)
        layer = kl.Dense(
            16, activation="sigmoid",
            activity_regularizer=regularizers.l2(0.005)
        )(lstm)
        lstm2 = kl.LSTM(
            8, activity_regularizer=regularizers.l2(0.01), recurrent_regularizer=regularizers.l2(0.001),
            dropout=0.2, recurrent_dropout=0.2
        )(layer)
        out = kl.Dense(
            1, activation="sigmoid",
            activity_regularizer=regularizers.l2(0.001)
        )(lstm2)

        model = Model(input_data, out)
        model.compile(optimizer="adam",loss="mean_squared_error", metrics=["mse"])

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
        test_y = np.array(pd.read_csv("features/autoencoded_test_y.csv", index_col=0))

        stock_data_test = np.array(pd.read_csv("stock_data_test.csv", index_col=0))

        print('Model MSE: {}'.format(model.evaluate(test_x, test_y)))
        stock_data = []
        for i in range(len(test_y)):
            prediction = model.predict(np.reshape(test_x[i], (1, 1, self.input_shape)))
            stock_price = np.exp(np.reshape(prediction, (1,))) * stock_data_test[i]
            stock_data.append(stock_price[0])

        stock_data[:] = [i - (float(stock_data[0]) - float(stock_data_test[0])) for i in stock_data]

        predicted_data = pd.DataFrame(stock_data)
        predicted_data[0].plot(label='Predicted Price', figsize=(
            16, 8), title='Prediction vs Actual')

        actual_data = pd.DataFrame(stock_data_test)
        actual_data[0].plot(label='Actual')
        
        stock = pd.DataFrame(stock_data, index=None)
        stock.to_csv("sample_predictions/AAPL_predicted_prices.csv")
        stock_test = pd.DataFrame(stock_data_test, index=None)
        stock_test.to_csv("sample_predictions/AAPL_actual_prices.csv")
        plt.legend()
        plt.show()

        price_r_score = r2_score(stock_data, stock_data_test[:-1])
        print('R^2 score {}'.format(price_r_score))
