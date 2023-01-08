import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score
import plotly.express as px
import base64
import hashlib


cols = ['predicted_bank', 'predicted_digital', 'customer_health_score', 'fintech_familiarity', 'top_spend_ecommerce_category', 
        'favorite_ecommerce_category']

def generate_opa_id(phone:str) -> str:
    hash_phone = hashlib.sha1(str.encode(phone)).hexdigest()
    opa_id = "67762" + hash_phone + phone[-4:]
    return opa_id

def generate_df(df):
    df_final = df.loc[:,["opa_id", "label", 'continuity', 'character', 'is_male']]
    for col in cols:
        df_dummy = pd.get_dummies(df[col], drop_first=True, prefix=col)
        df_final = pd.concat([df_final, df_dummy], axis=1)
    attributes = [x for x in df_final.columns if x not in ["opa_id", "label"]]
    return df_final, attributes

def generate_df_all(df):
    df_final = df.loc[:,["opa_id", "label", 'continuity', 'character', 'is_male']]
    for col in cols:
        df_dummy = pd.get_dummies(df[col], prefix=col)
        df_final = pd.concat([df_final, df_dummy], axis=1)
    return df_final

def fit_PU_estimator(X,y, hold_out_ratio, estimator):
    
    # find the indices of the positive/labeled elements
    assert (type(y) == np.ndarray), "Must pass np.ndarray rather than list as y"
    positives = np.where(y == 1.)[0] 
    # hold_out_size = the *number* of positives/labeled samples 
    # that we will use later to estimate P(s=1|y=1)
    hold_out_size = int(np.ceil(len(positives) * hold_out_ratio))
    np.random.shuffle(positives)
    # hold_out = the *indices* of the positive elements 
    # that we will later use  to estimate P(s=1|y=1)
    hold_out = positives[:hold_out_size]
    # the actual positive *elements* that we will keep aside
    X_hold_out = X[hold_out] 
    # remove the held out elements from X and y
    X = np.delete(X, hold_out,0) 
    y = np.delete(y, hold_out)
    # We fit the estimator on the unlabeled samples + (part of the) positive and labeled ones.
    # In order to estimate P(s=1|X) or  what is the probablity that an element is *labeled*
    estimator.fit(X, y)
    # We then use the estimator for prediction of the positive held-out set 
    # in order to estimate P(s=1|y=1)
    hold_out_predictions = estimator.predict_proba(X_hold_out)
    #take the probability that it is 1
    hold_out_predictions = hold_out_predictions[:,1]
    # save the mean probability 
    c = np.median(hold_out_predictions)
    return estimator, c

def predict_PU_prob(X, estimator, prob_s1y1):
    predicted_s = estimator.predict_proba(X)
    predicted_s = predicted_s[:,1]
    return predicted_s / prob_s1y1

def create_bin(x):
    return int(np.ceil(x/0.1))

def create_df_lift(y_positive, y_probs_adj):
    df_lift = pd.DataFrame()
    df_lift['similarity'] = y_probs_adj
    df_lift['label'] = y_positive
    df_lift['decile_rank'] = df_lift.similarity.apply(create_bin)
    df_lift_grouped = df_lift.groupby("decile_rank").agg({"label":['count', sum]})
    df_lift_grouped = df_lift_grouped.iloc[::-1]
    df_lift_grouped.columns = ['count_', 'sum_']
    total = sum(df_lift_grouped.count_)
    total_crossed = sum(df_lift_grouped.sum_)
    df_lift_grouped['perc_cum'] = df_lift_grouped.count_ / total
    df_lift_grouped['crossed_rate'] = df_lift_grouped.sum_ / df_lift_grouped.count_
    df_lift_grouped['cum_count'] = df_lift_grouped.count_.cumsum()
    df_lift_grouped['perc_cum_count'] = df_lift_grouped.cum_count / total
    df_lift_grouped['perc_event'] = df_lift_grouped.sum_ / total_crossed
    df_lift_grouped['gain'] = df_lift_grouped.perc_event.cumsum()
    df_lift_grouped['cum_lift'] = df_lift_grouped.gain / df_lift_grouped.perc_cum_count
    return df_lift_grouped

def plot_bar(col, df):
    return px.bar(df, x=col, y="counts", color="counts", text="counts", title=f"Percentage of {col} from Crossed User", color_continuous_scale="blues", labels={
                     "counts": "Percentage(%)"
                 })

def lift_reach_plot(df):
    return px.line(df, x='perc_cum_count', y='cum_lift', markers=True, title=f"Uplift-Reach", labels={
                     "perc_cum_count": "Reach",
                     "cum_lift": "Lift"
                 })

def prec_recall_plot(precision, recall):
    return px.line(x=precision, y=recall, markers=True, title=f"Precision-Recall", labels={
                     "x": "Precision",
                     "y": "Recall"
                 })

def get_precision_recall(y_positive, y_probs):
    precision = []
    recall = []
    thresholds = []
    for i in np.linspace(0,0.9,10):
        thres = round(i,2)
        thresholds.append(thres)
        y_predict = [1 if x >= thres else 0 for x in y_probs]
        precision.append(precision_score(y_positive, y_predict))
        recall.append(recall_score(y_positive, y_predict))
    return precision, recall

def download_button(object_to_download, download_filename, size):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv,
    Returns:
    -------
    (str): the anchor tag to download object_to_download
    """
    if isinstance(object_to_download, pd.DataFrame):
        object_to_download = object_to_download.loc[:,["opa_id"]].head(size).to_csv(index=False).encode('utf-8')

    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    dl_link = f"""
    <html>
    <head>
    <title>Start Auto Download file</title>
    <script src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
    <script>
    $('<a href="data:text/csv;base64,{b64}" download="{download_filename}">')[0].click()
    </script>
    </head>
    </html>
    """
    return dl_link

def get_figure(subplots, left, right):
    figure1_traces = []
    figure2_traces = []
    for trace in range(len(left["data"])):
        figure1_traces.append(left["data"][trace])
    for traces in figure1_traces:
        subplots.append_trace(traces, row=1, col=1)
    subplots['layout']['xaxis']['title'] = 'Precision'
    subplots['layout']['yaxis']['title'] = 'Recall'

    for trace in range(len(right["data"])):
        figure2_traces.append(right["data"][trace])
    for traces in figure2_traces:
        subplots.append_trace(traces, row=1, col=2)
    subplots['layout']['xaxis2']['title'] = 'Reach'
    subplots['layout']['yaxis2']['title'] = 'Uplift'
    return subplots