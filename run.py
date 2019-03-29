# %%
from preprocessing import PreProcessing
from data_retriever import DataRetrieverYahoo
from autoencoder import AutoEncoder
from data_processing import DataProcessing
from model import NeuralNetwork
from model_20_encoded import nnmodel

# %%
SPLIT = 0.8
FEATURE_SPLIT = 0.25
INPUT_DIM = 20
retriever = DataRetrieverYahoo("AAPL", "2000-01-01", "2019-03-21")
retriever.get_stock_data()
retriever.display_data()

# %%
preprocess = PreProcessing(SPLIT, FEATURE_SPLIT)
# %%
preprocess.make_wavelet_train()
preprocess.make_test_data()
# %%
autoencoder = AutoEncoder(INPUT_DIM)
# %%
autoencoder.build_train_model(input_shape=55, encoded1_shape=40, encoded2_shape=30, decoded1_shape=30, decoded2_shape=40)
# %%

process = DataProcessing(SPLIT, FEATURE_SPLIT)
# %%
process.make_train_data()
# %%
process.make_train_y()
# %%
process.make_test_data()
# %%
process.make_test_y()
# %%
model = NeuralNetwork(INPUT_DIM, True)
model.make_train_model()



dataset, average, std = nnmodel(10, 0.01, 0.01)
print(
    f"Price Accuracy Average = {average} \nPrice Accuracy Standard Deviation = {std}")
