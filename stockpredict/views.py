import imp
from venv import create
from django.http import JsonResponse
from matplotlib import scale
from .datasets import get_stock_data, load_dataset
from . stock_prediction_algo import scale_data, create_dataset, create_model, predict_future_prices
import json
from django.views.decorators.csrf import csrf_exempt


@csrf_exempt
def predictStockMarketPrice(req):
    if req.method == 'POST':
        reqBody = json.loads(req.body)
        stock_name = reqBody['stock_name']
        time_stamp = reqBody['time_stamp']
        # get_stock_data(stock_name, "1653868800")
        get_stock_data(stock_name, time_stamp)
        dataset = load_dataset(stock_name)
        df1 = dataset.reset_index()['c']
        predicted_data = scale_data(df1, stock_name)
        predicted_price = []
        for element in predicted_data:
            predicted_price.append(element[0])

        print(predicted_price)

        return JsonResponse({"message": "successfully predicted price", "payload": predicted_price}, safe=False, status=200)

    return JsonResponse({
        "message": "Method Not Allowed",
    }, safe=False, status=405)
