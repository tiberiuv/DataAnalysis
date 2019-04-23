# %%
import pandas_datareader.data as pdr
import fix_yahoo_finance as fix
fix.pdr_override()


class DataRetriever:
    def __init__(self, symbol, start, end):
        self.symbol = symbol
        self.start = start
        self.end = end

    # get stock data
    def get_stock_data(self):
        raise NotImplementedError

    def display_data(self):
        print(self.data.head())


class DataRetrieverYahoo(DataRetriever):
    def get_stock_data(self):
        data = pdr.get_data_yahoo(self.symbol, self.start, self.end)
        data.to_csv("stock_data.csv")
        self.data = data
        # print(data)
        return data


if __name__ == "__main__":
    retriever = DataRetrieverYahoo("AAPL", "2000-01-01", "2019-03-21")
    retriever.get_stock_data()
