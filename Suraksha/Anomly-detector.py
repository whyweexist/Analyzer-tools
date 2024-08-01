import argparse
import arviz as az
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly.graph_objects as go
import yfinance as yf

snp_https_path = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
start_date = '2022-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
output_dir = os.path.join('output/',end_date+'/')
data_filename = 'stock_prices.csv'

cache = {}
data_name = 'stock_data'
date_name = 'Date'

random_seed = 42

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--create_data',action='store_true')
    parser.add_argument('--read_data',action='store_true')
    parser.add_argument('--random_buy_sell_simulation',action='store_true')
    parser.add_argument('--quantile_outlier',action='store_true')
    parser.add_argument('--pymc_autoregressive_model',action='store_true')
    args = parser.parse_args()
    return args

def print_header(message):
    print('-'*100)
    print(message)
    print('-'*100)

def create_data():
    print_header('create_data')
    tickers = pd.read_html(snp_https_path)[0]
    df = yf.download(tickers.Symbol.to_list(),start_date,end_date,auto_adjust=True)['Close']
    os.makedirs(output_dir,exist_ok=True)
    df.to_csv(os.path.join(output_dir,data_filename))
    cache[data_name] = df

def read_data():
    print_header('read_data')
    cache[data_name] = pd.read_csv(os.path.join(output_dir,data_filename))

def plot_with_confidence_interval(xs,ys,ys_upper,ys_lower,this_output_dir,plotname,width=1000,height=500,style='band'):
    os.makedirs(this_output_dir,exist_ok=True)
    fig = go.Figure()
    fig.update_layout(
        autosize=False,
        width=width,
        height=height,
        paper_bgcolor="LightSteelBlue",
        title=dict(text=plotname),
    )
    if style == 'band':
        fig.add_traces([
            go.Scatter(
                x=xs,
                y=ys,
                mode='markers',
                showlegend=False,
            ),
            go.Scatter(
                x=xs,
                y=ys_upper,
                mode='lines',
                line_color='rgba(0,0,0,0)',
                showlegend=False,
            ),
            go.Scatter(
                x=xs,
                y=ys_lower,
                mode='lines', 
                line_color='rgba(0,0,0,0)',
                fill='tonexty', 
                fillcolor='rgba(0,0,255,0.2)',
                showlegend=False,
            )
        ])
    elif style == 'errorbar':
        fig.add_traces([
            go.Scatter(
                x=xs,
                y=ys,
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[ys_upper[i]-ys[i] for i in range(len(ys))],
                    arrayminus=[ys[i]-ys_lower[i] for i in range(len(ys))]
                )
            ),
        ])
    fig.write_image(os.path.join(this_output_dir,plotname+'.png'))

def quantile_outlier(x,y,plotname='plot',delta=100,alpha=0.005):
    print_header('quantile_outlier')
    n = len(y)
    xs,ys,ys_upper,ys_lower = [],[],[],[]
    for i in range(delta,n):
        delta_y = y[(i-delta):i]
        low = np.quantile(delta_y,q=alpha)
        high = np.quantile(delta_y,q=1.-alpha)
        xs.append(x[i])
        ys.append(y[i])
        ys_upper.append(high)
        ys_lower.append(low)
    
    this_output_dir = os.path.join(output_dir,'quantile_outlier/')
    plot_with_confidence_interval(xs,ys,ys_upper,ys_lower,this_output_dir,plotname)

def random_buy_sell_simulation_per_y(y,nboot=1000):
    n = len(y)
    average_profit = 0
    ys = []
    for i in range(nboot):
        buy_index = np.random.randint(n)
        sell_index = np.random.randint(n-buy_index)
        ys.append(y[sell_index]-y[buy_index])
    return ys

def random_buy_sell_simulation(alpha=0.005,plotname='average_profit'):
    xs,ys,ys_upper,ys_lower = [],[],[],[]
    for column in cache[data_name]:
        if column == date_name: continue
        y = cache[data_name][column]
        ys_per_stock = random_buy_sell_simulation_per_y(y)
        xs.append(column)
        ys.append(np.mean(ys_per_stock))
        ys_upper.append(np.quantile(ys_per_stock,q=1-alpha))
        ys_lower.append(np.quantile(ys_per_stock,q=alpha))
        if np.quantile(ys_per_stock,q=1-alpha) < 0. or np.quantile(ys_per_stock,q=alpha) > 0.:
            print('stock {:s} is not compatible with null hypothesis')
    this_output_dir = os.path.join(output_dir,'random_buy_sell_simulation/')
    plot_with_confidence_interval(xs,ys,ys_upper,ys_lower,this_output_dir,plotname,width=4000,style='errorbar')

def pymc_autoregressive_model(y):
    print_header('pymc_autoregressive_model')
    import pymc as pm
    with pm.Model() as ar:
        rho = pm.Normal("rho", mu=0.0, sigma=1.0, shape=2)
        tau = pm.Exponential("tau", lam=0.5)
        likelihood = pm.AR(
            "y", rho=rho, tau=tau, constant=True, init_dist=pm.Normal.dist(0, 10), observed=y
        )
        idata = pm.sample(100, tune=200, random_seed=random_seed)

        this_output_dir = os.path.join(output_dir,'pymc_autoregressive_model/')
        
        idata = pm.sample_posterior_predictive(idata, var_names=['y'], predictions=True)
        fig,ax = plt.subplots()
        ax.plot(idata.predictions['y'].isel(chain=0,draw=0),label='prediction')
        ax.plot(y,label='data')
        ax.legend()
        ax.grid()
        fig.savefig(os.path.join(this_output_dir,'forecast.png'))

def main():
    args = parse_arguments()
    if args.create_data: create_data()
    if args.read_data: read_data()
    if args.random_buy_sell_simulation: random_buy_sell_simulation()

    if args.quantile_outlier:
        for column in cache[data_name]:
            if column == date_name: continue
            x = cache[data_name][date_name]
            y = cache[data_name][column]
            quantile_outlier(x,y,column)

    if args.pymc_autoregressive_model:
        for column in cache[data_name]:
            if column == date_name: continue
            y = cache[data_name][column]
            pymc_autoregressive_model(y)

if __name__ == '__main__':
    main()
