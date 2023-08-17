from django.shortcuts import render
import talib
import numpy as np
import pandas as pd
import yfinance as yf
from yfinance import Ticker
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_curve, auc
import random
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn as nn
import json


# Create your views here.
def homepage(request):
    return render(request, 'homepage.html')


def svc(request):
    return render(request, 'svc.html')


def lstm(request):
    return render(request, 'lstm.html')


def random_forest(request):
    return render(request, 'random_forest.html')


def svc_prediction(request):
    C = request.POST.get('C', 1.0)
    gamma = request.POST.get('gamma', 'scale')
    stock_code = request.POST.get('stock_code')
    days = int(request.POST.get('days'))
    try:
        C = float(C)
    except ValueError:
        C = 1.0

    try:
        gamma = float(gamma)
    except ValueError:
        gamma = 'scale'

    try:
        org_data = yf.download(f'{stock_code}', period='5y').dropna()
    except Exception as e:
        error_message = str(e)
        if "n_samples=0" in error_message:
            result = "Stock code does not exist."
        else:
            result = "Please try again."
        return render(request, 'svc.html', {'result': result})

    dataset = dataProcessing(org_data, days)
    prediction, report, roc_auc, fpr, tpr = svc_predict(dataset, C=C, gamma=gamma)
    pred = prediction[-1]
    submit_message = "submitted successfully"
    report = [line.split() for line in report.strip().split('\n')]
    json_fpr = json.dumps(fpr.tolist())
    json_tpr = json.dumps(tpr.tolist())
    return render(request, 'svc.html', {'result': submit_message,
                                        'pred': pred,
                                        'report': report,
                                        'stock_code': stock_code,
                                        'days': days,
                                        'roc_auc': roc_auc,
                                        'json_fpr': json_fpr,
                                        'json_tpr': json_tpr
                                        })


def random_forest_prediction(request):
    n_jobs = request.POST.get('n_jobs', -1)
    n_estimators = request.POST.get('n_estimators', '65')
    stock_code = request.POST.get('stock_code')
    days = int(request.POST.get('days'))
    try:
        n_jobs = int(n_jobs)
    except ValueError:
        n_jobs = -1

    try:
        n_estimators = int(n_estimators)
    except ValueError:
        n_estimators = 65

    try:
        org_data = yf.download(f'{stock_code}', period='5y').dropna()
    except ValueError:
        result = f"Stock code '{stock_code}' does not exist."
        return render(request, 'random_forest.html', {'result': result})

    dataset = dataProcessing(org_data, days)
    prediction, report, roc_auc, fpr, tpr = random_forest_predict(dataset, n_jobs=n_jobs, n_estimators=n_estimators)
    pred = prediction[-1]
    submit_message = "submitted successfully"
    report = [line.split() for line in report.strip().split('\n')]
    json_fpr = json.dumps(fpr.tolist())
    json_tpr = json.dumps(tpr.tolist())
    return render(request, 'random_forest.html', {'result': submit_message,
                                                  'pred': pred,
                                                  'report': report,
                                                  'stock_code': stock_code,
                                                  'days': days,
                                                  'roc_auc': roc_auc,
                                                  'json_fpr': json_fpr,
                                                  'json_tpr': json_tpr
                                                  })


def lstm_prediction(request):
    stock_code = request.POST.get('stock_code')
    days = int(request.POST.get('days'))
    hidden_dim = request.POST.get('hidden_dim')
    num_epochs = request.POST.get('num_epochs')
    try:
        hidden_dim = int(hidden_dim)
    except ValueError:
        hidden_dim = 32

    try:
        num_epochs = int(num_epochs)
    except ValueError:
        num_epochs = 100

    try:
        org_data = yf.download(f'{stock_code}', period='5y').dropna()
    except ValueError:
        result = f"Stock code '{stock_code}' does not exist."
        return render(request, 'lstm.html', {'result': result})

    dataset = dataProcessing(org_data, days)
    prediction, report, roc_auc, fpr, tpr = lstm_predict(dataset, hidden_dim=hidden_dim, num_epochs=num_epochs)
    pred = prediction[-1]
    submit_message = "submitted successfully"
    report = [line.split() for line in report.strip().split('\n')]
    json_fpr = json.dumps(fpr.tolist())
    json_tpr = json.dumps(tpr.tolist())
    return render(request, 'lstm.html', {'result': submit_message,
                                         'pred': pred,
                                         'report': report,
                                         'stock_code': stock_code,
                                         'days': days,
                                         'roc_auc': roc_auc,
                                         'json_fpr': json_fpr,
                                         'json_tpr': json_tpr
                                         })


def random_forest_predict(dataset, n_jobs=-1, n_estimators=65, random_state=5, SPLIT_RATIO=0.7, ):
    np.random.seed(5)
    random.seed(5)
    x = dataset[[x for x in dataset.columns if x not in ['pred']]]
    y = dataset['pred']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=SPLIT_RATIO, random_state=5)
    pipe = RandomForestClassifier(n_jobs=n_jobs, n_estimators=n_estimators, random_state=random_state)
    pipe.fit(x_train, y_train.values.ravel())
    prediction = pipe.predict(x_test)
    report = classification_report(y_test, prediction)
    probs = pipe.predict_proba(x_test)[:, 1]  # Probability values predicted by the model
    fpr, tpr, _ = roc_curve(y_test, probs)  # Calculate the points of the ROC curve
    roc_auc = auc(fpr, tpr)  # Calculate AUC
    # Using all data sets to make predictions in the next n days
    full_prediction = pipe.predict(x)
    return full_prediction, report, roc_auc, fpr, tpr


def svc_predict(data, C=1.0, gamma='scale', cache_size=200, class_weight='balanced', SPLIT_RATIO=0.7,
                probability=True):
    # the default parameters are set to the best parameters since the test
    # class_weight could be None or balanced
    # Probability values must be true to output the probability, which in turn calculates ROC and AUC
    x = data[[col for col in data.columns if col != 'pred']]
    y = data['pred']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=SPLIT_RATIO)
    pipe = Pipeline([
        ('scl', StandardScaler()),
        ('clf', SVC(C=C, gamma=gamma, cache_size=cache_size, probability=probability, class_weight=class_weight))])
    pipe.fit(x_train, y_train.values.ravel())
    # Using test sets to make predictions and generate reports and Roc diagram parameters
    prediction = pipe.predict(x_test)
    report = classification_report(y_test, prediction)
    probs = pipe.predict_proba(x_test)[:, 1]  # Probability values predicted by the model
    fpr, tpr, _ = roc_curve(y_test, probs)  # Calculate the points of the ROC curve
    roc_auc = auc(fpr, tpr)  # Calculate AUC
    # Using all data sets to make predictions in the next n days
    full_prediction = pipe.predict(x)
    return full_prediction, report, roc_auc, fpr, tpr


def dataProcessing(org_data, n):
    # Smoothing index weighting，Call ewm to implement，Increase temporal relevance and reduce fallout
    s_data = org_data.ewm(alpha=0.9).mean()
    # add Technical indicators
    data = featureExtensions(s_data).dropna().iloc[:-n]
    # generate predict column
    pred = (s_data.shift(-n)['Close'].values >= s_data['Close'])
    # Remove the first n values that are not computed and Converts Boolean values to integers,
    # False converts to 0 and True converts to 1.
    pred = pred.iloc[:-n].astype(int)
    data['pred'] = pred
    del (data['Close'])  # delete the close price to preventing predicted values from tracking to it
    return data.dropna()


def dataset_generation_for_LSTM(x, pred, Impact_length=30, Split_ratio=0.7):
    num_features = x.shape[1]
    pred_array = pred.values.reshape(-1, 1)
    data_length = len(pred_array)
    selected_data = x
    count_start = len(selected_data) - data_length
    temp = []

    for i in range(data_length):
        start_index = count_start - Impact_length
        end_index = count_start
        temp.append(selected_data[start_index:end_index].tolist())
        count_start += 1

    # Padding sequences to ensure they all have the same length
    max_length = max(len(seq) for seq in temp)
    padded_temp = [seq + [[0] * num_features] * (max_length - len(seq)) for seq in temp]
    Features = np.array(padded_temp)  # Convert to regular NumPy array
    Label = np.array(pred_array)

    x_train, x_test, y_train, y_test = train_test_split(
        Features, Label, test_size=1 - Split_ratio, random_state=500)

    return x_train, x_test, y_train, y_test, Features


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layer, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, output_dim)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layer, x.size(0), self.hidden_dim).requires_grad_()
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc1(out[:, -1, :])
        out = self.Sigmoid(out)
        return out


def lstm_predict(dataset, hidden_dim=32, num_epochs=100, num_layer=2, output_dim=1, Impact_length=30):
    # Except that the prediction columns are all treated as X
    x = dataset[[x for x in dataset.columns if x not in ['pred']]]
    y = dataset['pred']
    scaler = MinMaxScaler(feature_range=(-1, 1))
    x = scaler.fit_transform(x)
    # Get the number of features
    x_train, x_test, y_train, y_test, alldata = dataset_generation_for_LSTM(x, y, Impact_length)
    x_train = torch.from_numpy(x_train).type(torch.Tensor)
    x_test = torch.from_numpy(x_test).type(torch.Tensor)
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    input_dim = x_train.shape[2]
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layer=num_layer, output_dim=output_dim)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for i in range(num_epochs):
        y_train_prediction = model(x_train)
        loss = criterion(y_train_prediction, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    y_test_prediction = torch.round(model(x_test))
    prediction = torch.round(model(x_test)).detach().numpy()
    report = classification_report(y_test, prediction, zero_division=1)
    y_test_prediction = model(x_test).detach().numpy()
    y_test_prediction = np.round(y_test_prediction)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_prediction)
    roc_auc = auc(fpr, tpr)

    # Using all datasets without prediction column to make predictions in the next n days
    alldata = torch.from_numpy(alldata).type(torch.Tensor)
    input_dim = alldata.shape[2]
    model_for_prediction = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, num_layer=num_layer,
                                output_dim=output_dim)

    with torch.no_grad():
        full_prediction = torch.round(model_for_prediction(alldata))
        full_prediction_rounded = full_prediction.detach().numpy()

    return full_prediction_rounded, report, roc_auc, fpr, tpr


def featureExtensions(data):
    for x in [9, 14, 21, 30, 50, 90]:
        # Calculate RSI (Relative Strength Index)
        data['RSI_' + str(x)] = talib.RSI(data['Close'], timeperiod=x)
        # Calculate ROC (Rate of Change)
        data['ROC_' + str(x)] = talib.ROC(data['Close'], timeperiod=x)
        # Calculate Stochastic oscillator %D %K
        slowk, slowd = talib.STOCH(data['High'], data['Low'], data['Close'], fastk_period=x, slowk_period=x,
                                   slowd_period=x)
        data['STOCH_SlowK_' + str(x)] = slowk
        data['STOCH_SlowD_' + str(x)] = slowd
        # Accumulation/Distribution
        data['ACCDIST_' + str(x)] = talib.AD(data['High'], data['Low'], data['Close'], data['Volume'])
        # On-balance Volume
        data['OBV_' + str(x)] = talib.OBV(data['Close'], data['Volume'])
        # Commodity Channel Index
        data['CCI_' + str(x)] = talib.CCI(data['High'], data['Low'], data['Close'], timeperiod=x)
        # Ease of Movement
        prev_high = data['High'].shift(1)
        prev_low = data['Low'].shift(1)
        eom = ((data['High'] + data['Low']) / 2 - (prev_high + prev_low) / 2) / (data['High'] - data['Low']) * data[
            'Volume']
        data['EOM_' + str(x)] = eom
        # Trix
        data['TRIX_' + str(x)] = talib.TRIX(data['Close'], timeperiod=x)
        # Momentum
        data['MOM_' + str(x)] = talib.MOM(data['Close'], timeperiod=x)
        # oney Flow Index and Ratio
        data['MFI_' + str(x)] = talib.MFI(data['High'], data['Low'], data['Close'], data['Volume'], timeperiod=x)
        # Add n-day exponential moving average to the data
        ema = talib.EMA(data['Close'], timeperiod=x)
        data['EMA_' + str(x)] = ema
    # add MACD indicator to data
    macd, signal, _ = talib.MACD(data['Close'], fastperiod=14, slowperiod=50, signalperiod=9)
    data['MACD'] = macd
    data['MACD_Signal'] = signal
    # delete ''Volume' ,'Open', 'High','Low' to avoid data coupling
    del (data['Volume'])
    del (data['Open'])
    del (data['High'])
    del (data['Low'])
    del (data['Adj Close'])
    return data
