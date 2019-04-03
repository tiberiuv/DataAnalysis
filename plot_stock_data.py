
from matplotlib.dates import DateFormatter, WeekdayLocator,\
    DayLocator, MONDAY
import pandas as pd
import datetime
import matplotlib.dates as mdates
from matplotlib.dates import date2num
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from mpl_finance import candlestick_ohlc


class Plot_stock_data:

    def __init__(self, data, title):
        self.data = data
        self.title = title

    def pandas_candlestick_ohlc(self, stick="day", otherseries=None):
        dat = self.data
        mondays = WeekdayLocator(MONDAY)        # major ticks on the mondays
        alldays = DayLocator()              # minor ticks on the days
        dayFormatter = DateFormatter('%d')      # e.g., 12

        # Create a new DataFrame which includes OHLC data for each period specified by stick input
        transdat = dat.loc[:, ["Open", "High", "Low", "Close"]]
        if (type(stick) == str):
            if stick == "day":
                plotdat = transdat
                stick = 1  # Used for plotting
            elif stick in ["week", "month", "year"]:
                if stick == "week":
                    transdat["week"] = pd.to_datetime(transdat.index).map(
                        lambda x: x.isocalendar()[1])  # Identify weeks
                elif stick == "month":
                    transdat["month"] = pd.to_datetime(transdat.index).map(
                        lambda x: x.month)  # Identify months
                transdat["year"] = pd.to_datetime(transdat.index).map(
                    lambda x: x.isocalendar()[0])  # Identify years
                # Group by year and other appropriate variable
                grouped = transdat.groupby(list(set(["year", stick])))
                # Create empty data frame containing what will be plotted
                plotdat = pd.DataFrame(
                    {"Open": [], "High": [], "Low": [], "Close": []})
                for name, group in grouped:
                    plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                           "High": max(group.High),
                                                           "Low": min(group.Low),
                                                           "Close": group.iloc[-1, 3]},
                                                          index=[group.index[0]]))
                if stick == "week":
                    stick = 5
                elif stick == "month":
                    stick = 30
                elif stick == "year":
                    stick = 365

        elif (type(stick) == int and stick >= 1):
            transdat["stick"] = [np.floor(i / stick)
                                 for i in range(len(transdat.index))]
            grouped = transdat.groupby("stick")
            # Create empty data frame containing what will be plotted
            plotdat = pd.DataFrame(
                {"Open": [], "High": [], "Low": [], "Close": []})
            for name, group in grouped:
                plotdat = plotdat.append(pd.DataFrame({"Open": group.iloc[0, 0],
                                                       "High": max(group.High),
                                                       "Low": min(group.Low),
                                                       "Close": group.iloc[-1, 3]},
                                                      index=[group.index[0]]))

        else:
            raise ValueError(
                'Valid inputs to argument "stick" include the strings "day", "week", "month", "year", or a positive integer')

        # Set plot parameters, including the axis object ax used for plotting
        fig, ax = plt.subplots()
        fig.subplots_adjust(bottom=0.2)
        if plotdat.index[-1] - plotdat.index[0] < pd.Timedelta('730 days'):
            weekFormatter = DateFormatter('%b %d')  # e.g., Jan 12
            ax.xaxis.set_major_locator(mondays)
            ax.xaxis.set_minor_locator(alldays)
        else:
            weekFormatter = DateFormatter('%b %d, %Y')
        ax.xaxis.set_major_formatter(weekFormatter)

        ax.grid(True)

        # Create the candelstick chart
        candlestick_ohlc(ax, list(zip(list(date2num(plotdat.index.tolist())), plotdat["Open"].tolist(), plotdat["High"].tolist(),
                                      plotdat["Low"].tolist(), plotdat["Close"].tolist())),
                         colorup="black", colordown="red", width=stick * .5)

        # Plot other series (such as moving averages) as lines
        if otherseries != None:
            if type(otherseries) != list:
                otherseries = [otherseries]
            dat.loc[:, otherseries].plot(ax=ax, lw=1.5, grid=True)

        ax.xaxis_date()
        ax.autoscale_view()
        plt.setp(plt.gca().get_xticklabels(),
                 rotation=45, horizontalalignment='right')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.title(self.title)
        figure(figsize=(16, 8))

        plt.savefig('figures/' + self.title + '.png')
        plt.show()

    def comp_stock(self, data1, data2, data3, data4):
        data1['Adj Close'].plot(label='MSFT', figsize=(
            16, 8), title='Adjusted CLosing')
        data2['Adj Close'].plot(label='AAPL')
        data3['Adj Close'].plot(label='FB')
        data4['Adj Close'].plot(label='GOOGLE')

        plt.legend()

        plt.savefig('figures/' + 'MSFT_AAPL_FB_GOOGLE' + '.png')
        plt.show()


if __name__ == "__main__":
    plotting_data = Plot_stock_data(data, "AAPL")
    plotting_data.pandas_candlestick_ohlc()
    plotting_data.comp_stock(data1, data2, data3, data4)
