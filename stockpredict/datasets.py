from fileinput import filename
import requests
import pandas as pd
import os
from django.conf import settings


# to get the stock price from nepsealpha
def get_stock_data(stock, timestamp):
    url = "https://www.nepsealpha.com/trading/1/history?symbol=" + stock + \
        "&resolution=1D&to=+" + timestamp + "&pass=ok&currencyCode=NRS"
    headers = {
        'Accept': 'application/json',
        'Content-Type': 'application/json'
    }
    response = requests.request('GET', url, headers=headers, data={})
    myJson = response.json()
    df = pd.DataFrame(myJson)
    filename = stock + ".csv"
    new_path = settings.MEDIA_ROOT + filename
    df.to_csv(new_path, index=False)


# to read the dataset from media directory
def load_dataset(stock_name):
    df = pd.read_csv(settings.MEDIA_ROOT + stock_name + ".csv")
    return df
