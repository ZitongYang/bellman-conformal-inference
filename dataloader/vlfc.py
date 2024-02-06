import warnings
warnings.simplefilter(action='ignore', category=DeprecationWarning)
import sys
sys.path.append('.') 

from utils import make_nps_chi2
from utils.function01 import SymFunction01
from dataloader import ForecastingData

import pandas as pd
import numpy as np
from arch import arch_model
from tqdm import tqdm

COMPANIES = ['AMD', 'Amazon', 'Nvidia']

def preprocess(cid):
    """
    Reading the raw stock data
    """
    df = pd.read_csv('data/raw/{}.csv'.format(cid))
    df.loc[:,'Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    df = df.sort_values(by='Date')

    # Volatility computation
    df.loc[:,'Pt'] = df[' Open']
    Ptp1 = np.array(df[' Open'][1:]) # Calculate the next day's opening price
    df = df[:-1].copy()
    df.loc[:,'Ptp1'] = Ptp1
    df.loc[:,'1e2Rt'] = 100*(df['Ptp1'] - df['Pt'])/df['Pt'] # Rt is the return
    df.loc[:,'1e2Vt'] = np.abs(df['1e2Rt'])

    # (Optional) Winsorization
    # df = df[df['1e2Vt'].between(np.quantile(df['1e2Vt'], 0.05),
    #                             np.quantile(df['1e2Vt'], 0.95))]

    df.loc[:,'1e4Vt2'] = df['1e2Vt']**2 # squared volatility
    df = df.set_index('Date')
    df = df[['1e2Vt', '1e4Vt2', '1e2Rt']]

    return df


def forecast(returns, T, start_date, curr_date):
    """
    Use the return information returns[start_date, curr_date] to estimate the
    (a) GARCH(1, 1) parameter on sigma^2 for
        curr_date+1, curr_date+2, ..., curr_date+T .
    (b) the estimate of mu
    """
    train = returns[start_date:curr_date][['1e2Rt']]
    garch11 = arch_model(train, p=1, o=0, q=1, dist='normal')
    res = garch11.fit(update_freq=100, disp='off')
    sigma2Kpt = res.forecast(horizon=T,
                             start=train.last_valid_index(),
                             reindex=False).variance
    sigma2Kpt = np.array(sigma2Kpt).reshape(-1)
    return sigma2Kpt, res.params['mu']


def make_forecasted_data(cid):
    df = preprocess(cid)
    dates = df.index

    m = 100 # length of fitting window: amount of past data to use
    horizon = 14 # length of forecasting into the future

    prev_nps = None
    for i in tqdm(range(m, len(df))):
        start_date = dates[i-m]
        end_date = dates[i-1]
        date = dates[i]
        sigma2hatKpt, muhat = forecast(df, horizon, start_date, end_date)
        df.loc[date, 'muhat'] = muhat
        for j in range(1, horizon+1):
            df.loc[date, 'sigma2_{}'.format(j)] = sigma2hatKpt[j-1]
        nps = make_nps_chi2(sigma2hatKpt, muhat)
        
        if prev_nps:
            true_Ytp1 = float(df.loc[end_date]['1e4Vt2'])
            beta = prev_nps[0].beta_threshold(true_Ytp1)
            df.loc[end_date, 'beta'] = beta
        prev_nps = nps
        
    df = df.dropna()
    df.to_csv('data/vlfc/{}-fc.csv'.format(cid))


class VolatilityData(ForecastingData):
    def __init__(self, cid, beta_cdf_len=100):
        super().__init__('vlfc', cid)
        
        # load forecasting df for the experiemnt
        df = pd.read_csv('data/vlfc/{}-fc.csv'.format(cid))
        df.loc[:,'Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.sort_values(by='Date')
        self.df = df.set_index('Date')

        # experiment setup
        self.dates = self.df.index
        self.beta_cdf_len = beta_cdf_len
        self.curr_ind = self.beta_cdf_len
        self.end_ind = len(self.df)-1
        self.name = 'vlfc-{}'.format(cid)

    def next(self):
        if self.curr_ind > self.end_ind:
            return False
        else:
            date = self.dates[self.curr_ind]
            true_Vt2 = float(self.df.loc[date]['1e4Vt2'])
    
            # forecasting data
            muhat = self.df.loc[date]['muhat']
            sigma_columns = [col for col in 
                             self.df.columns if col.startswith('sigma2_')]
            sigma_columns.sort(key=lambda x: int(x.split('_')[1]))
            sigma2hatKpt = np.array(self.df.loc[date, sigma_columns].values)
    
            # make nps
            nps = make_nps_chi2(sigma2hatKpt, muhat)
    
            # make empirical beta dist
            start_date = self.dates[self.curr_ind- self.beta_cdf_len]
            end_date = self.dates[self.curr_ind- 1]
            betas = np.array(self.df[start_date:end_date][['beta']]).reshape(-1)
            beta_cdf = SymFunction01('np.mean(betas.reshape(-1, 1)<alpha, axis=0)',
                                     'alpha', {'betas': betas})
    
            # update curr_ind
            self.curr_ind += 1
            return {'true_y': true_Vt2,
                    'beta': self.df.loc[date]['beta'],
                    'nested_pred_sets': nps,
                    'beta_cdf': beta_cdf,
                    'index': date}

    def refresh(self):
        self.curr_ind = self.beta_cdf_len

    def expectancy(self):
        return self.end_ind - self.curr_ind


if __name__ == '__main__':
    # make_forecasted_data('Amazon')
    # for cid in COMPANIES:
    # make_forecasted_data(cid)
    exp = VolatilityData('Amazon')
    exp.plot_ecc()