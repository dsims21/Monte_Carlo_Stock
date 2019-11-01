import numpy as np
import pandas as pd
#from pandas_datareader import data as wb
#import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats
#import matplotlib.mlab as mlab
#from pandas.tseries.holiday import USFederalHolidayCalendar, USColumbusDay
#from pandas.tseries.offsets import CustomBusinessDay
#import gzip
#import awscli
import boto3
#import csv
from pyathena import connect
import re
from cStringIO import StringIO
import datetime as dt

from pandas.tseries.holiday import AbstractHolidayCalendar, Holiday, nearest_workday, \
    USMartinLutherKingJr, USPresidentsDay, GoodFriday, USMemorialDay, \
    USLaborDay, USThanksgivingDay

class USTradingCalendar(AbstractHolidayCalendar):
    rules = [
        Holiday('NewYearsDay', month=1, day=1, observance=nearest_workday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday('USIndependenceDay', month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday('Christmas', month=12, day=25, observance=nearest_workday)
    ]

def get_trading_close_holidays(year):
    inst = USTradingCalendar()

    return inst.holidays(dt.datetime(year-1, 12, 31), dt.datetime(year, 12, 31))

def get_minutes_until_expiration(start, end): # Inclusive, Exclusive (2/5 to 2/8 [fri] yields 3, but 2/5 to 2/9 [sat] yields 4) and sat doesnt count here.
    # The exclusive nature of the end date might be confusing
    # It could be planned for here - and the variable name changed to exp.
    # CBOE has expiration dates on Fridays
    # This function will need them to be on the true Sat exp.

    inst = USTradingCalendar()

    startdate = dt.date.today() - dt.timedelta(days=365)  # Start the list a year ago
    enddate = dt.date.today() + dt.timedelta(days=1825)  # End the list 5 years from now
    mktholidays = np.array(inst.holidays(start=startdate, end=enddate), dtype='datetime64[D]').tolist()

    startcount = start
    endcount = end

    busdays = np.busday_count(startcount, endcount, holidays=mktholidays)
    total_mins = busdays * 390
    return total_mins


def main():

    # Create a data frame for final values to sit
    outputdata = pd.DataFrame(columns=['underlying_symbol',
                                       'quote_date',
                                       'last_close',
                                       'model_day_count',
                                       'expiration',
                                       'minutes_to_exp',
                                       'iterations',
                                       'mean',
                                       'stddev',
                                       'strike',
                                       'cdf_above_strike',
                                       'cdf_below_strike'])
    outputdata

    lastdate = list()
    eodlastdate = list()

    #Define S3 Connector
    s3 = boto3.resource('s3',
                        aws_access_key_id='xxx',
                        aws_secret_access_key='xxx')

    # Heres the regex for the date ([^sec_]*)(?=.csv)
    regex = re.compile(r'([^sec_]*)(?=.csv)')
    s3list = pd.DataFrame()

    #Create a list of objects in directory. (Find the date of the third to last)
    my_bucket = s3.Bucket('underlying-equity-interval')
    for object in my_bucket.objects.all():
        selectedfiles = re.search(regex,str(object.key))
        if selectedfiles:
            stringtoadd = str(object.key).split("sec_")[1].split(".csv")[0]
            print(stringtoadd)
            lastdate.append(stringtoadd)
            lastdate

    # This is the last day of files.
    filedate = lastdate[-1]

    calculation_days = 3 #How many days back should calculation go?
    queryday = lastdate[-calculation_days] #This gives the third day from the last, which can be inserted into the query.

    #To get the last date of the files in EOD Options...
    eodregex = re.compile(r'([^EODCalcs_]*)(?=.csv)')

    eodcalcs_bucket = s3.Bucket('underlying-options-eod-with-calcs-partitioned')
    for object in eodcalcs_bucket.objects.all():
        selectedeodfiles = re.search(eodregex, str(object.key))
        if selectedeodfiles:
            eodstringtoadd = str(object.key).split("EODCalcs_")[1].split(".csv")[0]
            print(eodstringtoadd)
            eodlastdate.append(eodstringtoadd)
            eodlastdate

    # This is the last day of files.
    eodfiledate = eodlastdate[-1]


    #Define Athena Connector
    conn = connect(aws_access_key_id='xxx',
                   aws_secret_access_key='xxx',
                   s3_staging_dir='s3://xxx/',
                   region_name='us-east-1')

    # New method where 'mid' (bid+ask)/2 is the measure column (retaining name 'close'for ease of integration)
    data = pd.read_sql(
        "SELECT underlying_symbol, quote_datetime, ((bid+ask)/2) as close FROM historical.equityinterval where dt >= cast('" + queryday + "' AS DATE) AND Date_Format(quote_datetime, '%T') <= date_format(timestamp '1900-01-01 16:00:00', '%T') AND (underlying_symbol IN ('AAPL', 'ACN', 'AGN', 'ALXN', 'AMGN', 'AMZN', 'AXP', 'BA', 'BABA', 'BIIB', 'BKNG', 'CAT', 'CELG', 'CHTR', 'CMG', 'COST', 'CRM', 'CVX', 'DATA', 'DE', 'DIA', 'DIS', 'DPZ', 'DRI', 'EA', 'EXPE', 'FB', 'FDX', 'FXE', 'GLD', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTU', 'ISRG', 'JNJ', 'JPM', 'LMT', 'LOW', 'MA', 'MAR', 'MCD', 'MMM', 'MSFT', 'NFLX', 'NOW', 'NSC', 'NVDA', 'ORLY', 'PANW', 'QQQ', 'RCL', 'RHT', 'RTN', 'SPG', 'SPY', 'TLT', 'TSLA', 'UNP', 'UPS', 'V', 'VMW', 'WDAY', 'WMT', 'WYNN', 'XOM', '^DJX', '^NDX', '^SPX', '^VIX'));", conn)

    # Old method where 'close' was measure column
    # data = pd.read_sql(
    #     "SELECT underlying_symbol, quote_datetime, close FROM historical.equityinterval where close != 0 and dt >= cast('" + queryday + "' AS DATE) AND Date_Format(quote_datetime, '%T') <= date_format(timestamp '1900-01-01 16:00:00', '%T') AND (underlying_symbol IN ('AAPL', 'ACN', 'AGN', 'ALXN', 'AMGN', 'AMZN', 'AXP', 'BA', 'BABA', 'BIIB', 'BKNG', 'CAT', 'CELG', 'CHTR', 'CMG', 'COST', 'CRM', 'CVX', 'DATA', 'DE', 'DIA', 'DIS', 'DPZ', 'DRI', 'EA', 'EXPE', 'FB', 'FDX', 'FXE', 'GLD', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTU', 'ISRG', 'JNJ', 'JPM', 'LMT', 'LOW', 'MA', 'MAR', 'MCD', 'MMM', 'MSFT', 'NFLX', 'NOW', 'NSC', 'NVDA', 'ORLY', 'PANW', 'QQQ', 'RCL', 'RHT', 'RTN', 'SPG', 'SPY', 'TLT', 'TSLA', 'UNP', 'UPS', 'V', 'VMW', 'WDAY', 'WMT', 'WYNN', 'XOM', '^DJX', '^NDX', '^SPX', '^VIX'));", conn)

    print(data.head())
    print(data.describe())
    print(data.dtypes)

    # Since ^VIX Starts at 9:32 (per CBOE), this will populate 9:32 data into 9:31 for the time being.
    mask = (data['underlying_symbol'].eq('^VIX') &
            data['quote_datetime'].dt.strftime('%H:%M:%S').eq('09:31:00') &
            data['close'].eq(0))

    data['close'] = data['close'].mask(mask, data['close'].shift(-1))

    data

    # data['quote_datetime'] = data['quote_datetime'].astype('str')
    # print(data.dtypes)

    #Pivot so that colums are symbols
    data = data.pivot(index="quote_datetime",columns="underlying_symbol",values="close")
    data

    #DROPS COLUMNS WHICH CONTAIN EVEN ONE NULL VALUE
    data = data.dropna(axis='columns')
    data

    #Get strikes for each symbol (Last available file in EODOptions is the source)
    strikes = pd.read_sql(
        "SELECT underlying_symbol, strike FROM historical.underlyingoptionseodwcalcs_partitioned where ((dt = CAST('"+ eodfiledate +"' AS DATE)) AND (underlying_symbol IN ('AAPL', 'ACN', 'AGN', 'ALXN', 'AMGN', 'AMZN', 'AXP', 'BA', 'BABA', 'BIIB', 'BKNG', 'CAT', 'CELG', 'CHTR', 'CMG', 'COST', 'CRM', 'CVX', 'DATA', 'DE', 'DIA', 'DIS', 'DPZ', 'DRI', 'EA', 'EXPE', 'FB', 'FDX', 'FXE', 'GLD', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTU', 'ISRG', 'JNJ', 'JPM', 'LMT', 'LOW', 'MA', 'MAR', 'MCD', 'MMM', 'MSFT', 'NFLX', 'NOW', 'NSC', 'NVDA', 'ORLY', 'PANW', 'QQQ', 'RCL', 'RHT', 'RTN', 'SPG', 'SPY', 'TLT', 'TSLA', 'UNP', 'UPS', 'V', 'VMW', 'WDAY', 'WMT', 'WYNN', 'XOM', '^DJX', '^NDX', '^SPX', '^VIX'))) GROUP BY underlying_symbol, strike Order by underlying_symbol, strike;",
        conn)
    strikes

    strikes["yindex"] = strikes.groupby("underlying_symbol").cumcount()
    new_df = strikes.pivot(index="yindex", columns="underlying_symbol", values="strike")
    strikes
    new_df

    #Get expirations for each symbol (Last available file in EODOptions is the source)
    expiries = pd.read_sql(
        "SELECT underlying_symbol, expiration FROM historical.underlyingoptionseodwcalcs_partitioned where ((dt = CAST('" + eodfiledate + "' AS DATE)) AND (underlying_symbol IN ('AAPL', 'ACN', 'AGN', 'ALXN', 'AMGN', 'AMZN', 'AXP', 'BA', 'BABA', 'BIIB', 'BKNG', 'CAT', 'CELG', 'CHTR', 'CMG', 'COST', 'CRM', 'CVX', 'DATA', 'DE', 'DIA', 'DIS', 'DPZ', 'DRI', 'EA', 'EXPE', 'FB', 'FDX', 'FXE', 'GLD', 'GOOG', 'GOOGL', 'GS', 'HD', 'HON', 'IBM', 'INTU', 'ISRG', 'JNJ', 'JPM', 'LMT', 'LOW', 'MA', 'MAR', 'MCD', 'MMM', 'MSFT', 'NFLX', 'NOW', 'NSC', 'NVDA', 'ORLY', 'PANW', 'QQQ', 'RCL', 'RHT', 'RTN', 'SPG', 'SPY', 'TLT', 'TSLA', 'UNP', 'UPS', 'V', 'VMW', 'WDAY', 'WMT', 'WYNN', 'XOM', '^DJX', '^NDX', '^SPX', '^VIX'))) GROUP BY underlying_symbol, expiration Order by underlying_symbol, expiration;",
        conn)
    expiries

    expiries["yindex"] = expiries.groupby("underlying_symbol").cumcount()
    expiries_new_df = expiries.pivot(index="yindex", columns="underlying_symbol", values="expiration")
    expiries
    expiries_new_df



    for key, value in data.iteritems():  #For Each Symbol
        print key, value
        print(key,value.describe()) #This is still weird - bringing back a count of all.

        #Show price plot
        #key,value.plot(figsize=(10, 6))  #used to be 'data' instead of key, value
        #plt.show()

        #Log Returns
        log_returns = np.log(1 + value.pct_change()) #used to be data
        log_returns.tail()

        print(log_returns.describe())


        #Remove specific rows (This is probably really inefficient right now)
        log_returns = log_returns.to_frame()
        log_returns
        print(log_returns.dtypes)
        log_returns = log_returns.reset_index()
        log_returns
        print(log_returns.dtypes)
        log_returns['quote_datetime'] = log_returns['quote_datetime'].astype('str')
        print(log_returns.dtypes)
        #test = test.drop(test[test["quote_datetime"].str.contains('09:31')], axis=0, inplace=True)
        log_returns
        print(log_returns.quote_datetime)
        log_returns =log_returns[~log_returns.quote_datetime.str.contains("09:31")]
        log_returns
        log_returns['quote_datetime'] = log_returns['quote_datetime'].astype('datetime64')
        print(log_returns.dtypes)
        log_returns = log_returns.set_index('quote_datetime')
        log_returns
        log_returns = pd.Series(log_returns[key].values,index=log_returns.index)
        log_returns
        #Should this be converted back to a series?
        #The problem here might be that the column isn't named properly.
        #It should look exactly like when it came in.


        #log_returns.plot(figsize=(10, 6))
        #plt.show()

        u = log_returns.mean()
        var = log_returns.var()
        drift = 0 # Used to be (u - (0.5 * var)), but with such small increments, drift is irrelevant
        stdev = log_returns.std()
        np.array(drift)
        drift    #used to be drift.values (Changed when data was swtiched to value during for loop implementation)
        stdev   #used to be stddev.values
        norm.ppf(0.95)   #What is this? Does it set some variable?
        x = np.random.rand(10, 2) #probably not needed. Gets overwritten even
        norm.ppf(x)   #probably not needed. Demonstration only.
        Z = norm.ppf(np.random.rand(10, 2)) #Probably not needed
        expiration = (dt.datetime.strptime(filedate, "%Y-%m-%d") + dt.timedelta(days=90)).strftime("%Y-%m-%d")
        t_intervals = get_minutes_until_expiration(filedate, expiration) #This should have a unit test.
        iterations = 10000
        daily_returns = np.exp(drift + stdev * norm.ppf(np.random.rand(t_intervals, iterations)))
        daily_returns
        S0 = value.iloc[-1]   #used to be data
        price_list = np.zeros_like(daily_returns)
        price_list[0]
        price_list[0] = S0
        for t in range(1, t_intervals):
            price_list[t] = price_list[t - 1] * daily_returns[t]
        price_list
        #plt.figure(figsize=(10, 6))
        #plt.plot(price_list);
        #plt.show()




        ##I think this is where the for loop should start. Provided that some arbitrary number is used for expiration above/
        ##this will allow for each expiration integer to be examined within the variable x below.

        #For each int > 180 days
        #  search index in pricelist - 1
        #  do stats on the distribution of the row (That slice in time)
        #  do CDF of each strike from that row. (that slice in time)
        #  Write each row to CSV variable.
        #  Go to the next expiry.

        # Some stocks will only have a few within that range, and some will have a dozen.

        strikelist = new_df[key]  # Just take the column of the current stock.
        strikelist = strikelist.dropna(axis=0, how='any')  # drop the NaNs
        strikelist

        expirieslist = expiries_new_df[key]  # Just take the column of the current stock.
        expirieslist = expirieslist.dropna(axis=0, how='any')  # drop the NaNs
        expirieslist

        for expiresrow in expirieslist: # For each expiration

                mins_to_exp = get_minutes_until_expiration(filedate,expiresrow)
                mins_to_exp

                # Needs an IF statement for excluding higher than day counts
                if mins_to_exp >= 0 and mins_to_exp <= t_intervals:
                    print(key)
                    print(expiresrow)
                    print(mins_to_exp)

                    # Create an array of only the last day.
                    x = price_list[mins_to_exp-1] ##Task - double verify that this is pulling the right row.

                    # Histogram of Prices
                    #num_bins = 150
                    #n, bins, patches = plt.hist(x, num_bins, facecolor='blue', alpha=0.5)
                    #plt.show()

                    # # Histogram of Returns
                    # R = np.zeros(10000)
                    # for i in range(10000):
                    #     R[i] = (x[i] - S0) / S0

                    #num_bins = 150
                    #n, bins, patches = plt.hist(R, num_bins, facecolor='blue', alpha=0.5)
                    #plt.show()

                    # PDF Statistics
                    mean = np.mean(x)
                    mean
                    stDev = np.std(x)
                    stDev

                    # # Test Statistics
                    # Rmean = np.mean(R)
                    # Rmean
                    # RstDev = np.std(R)
                    # RstDev

                    # #CDF Function example
                    # possibility = 1 - stats.norm(mean, stDev).cdf(275)
                    # possibility



                    for row in strikelist.iteritems(): #For each strike, perform CDF function.
                        normal_cdf_above = 1 - stats.norm(mean, stDev).cdf(row[1])
                        normal_cdf_below = stats.norm(mean, stDev).cdf(row[1])
                        outputdata = outputdata.append({'underlying_symbol': key,
                                                        'quote_date': filedate,
                                                        'last_close': S0,
                                                        'model_day_count': calculation_days,
                                                        'expiration': expiresrow,
                                                        'minutes_to_exp': mins_to_exp,
                                                        'iterations': iterations,
                                                        'mean': mean,
                                                        'stddev': stDev,
                                                        'strike': row[1],
                                                        'cdf_above_strike': normal_cdf_above,
                                                        'cdf_below_strike': normal_cdf_below},
                                                       ignore_index=True)
                        outputdata


    outputdata

    #Create a CSV and Send to S3
    csv_buffer = StringIO()
    outputdata.to_csv(csv_buffer)
    s3.Object('monte-carlo-results', 'dt='+filedate+'/'+'MonteCarloResults_'+filedate+'.csv').put(Body=csv_buffer.getvalue())

    #Trying this.
    # buffer = StringIO()
    # outputdata.to_csv(buffer)
    # buffer.seek(0)
    # obj = s3.Object('equity-interval-test', 'dt=' + filedate + '/' + 'MonteCarloResults_' + filedate + '.csv.gz')
    # obj.put(Body=buffer)
    #s3.put_object(buffer, Bucket='equity-interval-test', Key='dt=' + filedate + '/' + 'MonteCarloResults_' + filedate + '.csv.gz')

    # #Create a CSV
    # outputdata.to_csv('MonteCarloResults_' + filedate + '.csv.gz', compression='gzip')
    # print("Done")
    #
    # #Send output to S3
    # obj = s3.Object('equity-interval-test', 'dt='+filedate+'/'+'MonteCarloResults_'+filedate+'.csv.gz')
    # obj.put(Body=open('MonteCarloResults_' + filedate + '.csv.gz', 'rb'))
    # print("Done")






main()
