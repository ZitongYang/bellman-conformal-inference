from utils.function01 import Function0T, ConstantFunction01, \
                             SumFunction01, MulFunction01, \
                             SymFunction01
from utils.nps import NestedPredictionInterval

import scipy.stats as spstats
from scipy.stats import norm
import numpy as np
import pandas as pd
from scipy.stats import ncx2
import pickle
import yaml
import itertools


############# save/load dataframe+param dictionary #############
def save_df(namestr, df):
    df.to_csv('result/dataframe/{}.csv'.format(namestr))


def save_pickle(namestr, dic):
    with open('result/pickle/{}.pickle'.format(namestr),
              'wb') as f:
        pickle.dump(dic, f, pickle.HIGHEST_PROTOCOL)


def read_exp(namestr):
    df = pd.read_csv('result/dataframe/{}.csv'.format(namestr))
    df.loc[:,'index'] = pd.to_datetime(df['index'], format='%Y-%m-%d')
    df = df.sort_values(by='index')
    df = df.set_index('index')

    with open('result/pickle/{}.pickle'.format(namestr),
              'rb') as f:
        params = pickle.load(f)        
    return df, params


def read_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)
    return data


def gen_namestr(params):
    if params['method']=='aci' or params['method']=='bci':
        namestr = '{}-{}-{}-gamma_{}-lambda0_{}'.format(
                    params['task'], params['method'], params['id'],
                    params['gamma'] if params['gamma'] else 'N.A.',
                    params['lambda_init'] if params['lambda_init'] else 'N.A.')
    elif params['method']=='fixed':
        namestr = '{}-{}-{}-gamma_{}-lambda0_{}'.format(
                    params['task'], params['method'], params['id'],
                    'NA', 'NA')
    else:
        raise NotImplementedError("Method not supported")
    
    return namestr

############# some helper functions #############
def positive_part(arr):
    """
    implements the function f(x) = max(x, 0)
    """
    return np.array(arr>0, dtype=float) * arr


def less_one_part(arr):
    """
    implements the function f(x) = min(x, 1)
    """
    return np.array(arr < 1, dtype=float) * arr


def soft_normal_interval(x):
    flag = np.array(x > 1e-2, dtype=float)
    x = flag*x + (1-flag)*1e-2
    return spstats.norm.ppf(1 - x / 2)


def Q_non_central_chi2(quantile, mu, sigmasq):
    """
    Given Z ~ N( mu, sigmasq ), return the quantile-th quantile of Z^2
    """
    sigma = np.sqrt(sigmasq)
    return ncx2.ppf(quantile, 1, (mu/sigma)**2, scale=sigma**2)


def match_rank(beta_arr, alpha):
    """
    percentage of beta_arr larger than alpha
    """
    return np.mean(beta_arr>=alpha)

def compute_avg_no_inf(array):
    array = array[np.isfinite(array)]
    return np.mean(array)


def compute_local_avg(target, ma_window):
    result = []
    for i in range(ma_window, len(target) - ma_window):
        sample_data = target[i - ma_window:i + ma_window]
        result.append(compute_avg_no_inf(sample_data))
    return np.array(result)


def compute_local_median(target, ma_window):
    result = []
    for i in range(ma_window, len(target)-ma_window):
        sample_dates = target[i - ma_window:i + ma_window]
        result.append(np.median(sample_dates))
    return np.array(result)


def trim(array, ma_window):
    length = len(array)
    assert length > 2*ma_window
    return array[ma_window:length-ma_window]

############# helper functions relating to Function01   #############
def argmin01(L, D, F, bins):
    """
    Given L, D, F: [0, 1] -> R, define a new function
        * f(rho, alpha) = L(alpha) + D(rho)*F(alpha) that [T] X [0, 1] --> R.
    Given f, define two new functions
        * J(rho) = min_alpha f(alpha, rho)
        * alpha_star(rho) = argmin_alpha f(alpha, rho)
    :param L: <Function01>  L(alpha)
    :param D: <Function0T>  D(rho)
    :param F: <Function01>  F(alpha)
    :param bins: <int> #discretization when minimizing w.r.t. alpha
    :return alpha_star: <Function0T> alpha_star(rho)
    :return J: <Function0T> J(rho)
    """
    T =D.T
    Js = np.zeros(T+1)
    alpha_stars = np.zeros(T+1)

    for rho in range(T+1):
        constant_D_func = ConstantFunction01(D.eval(rho))
        f = SumFunction01(L, MulFunction01(constant_D_func, F))
        alpha_stars[rho], Js[rho] = f.min(bins)

    alpha_star = Function0T(T, alpha_stars)
    J = Function0T(T, Js)
    return alpha_star, J


############# generating nested prediction sets   #############
def make_nps_chi2(sigma2hatKpt, muhat):
    T = len(sigma2hatKpt)
    nps_lst = [NestedPredictionInterval(None, None) for _ in range(T)]
    for t in range(T):
        upper_func = SymFunction01('Q_non_central_chi2(1-alpha/2, mu, sigmasq)',
                                   'alpha',
                                   {'mu': muhat,
                                    'sigmasq': sigma2hatKpt[t],
                                    'Q_non_central_chi2': Q_non_central_chi2})
        lower_func = SymFunction01('Q_non_central_chi2(alpha/2, mu, sigmasq)',
                                   'alpha',
                                   {'mu': muhat,
                                    'sigmasq': sigma2hatKpt[t],
                                    'Q_non_central_chi2': Q_non_central_chi2})
        nps_lst[t] = NestedPredictionInterval(upper_func, lower_func)
    return nps_lst

def make_nested_pred_sets_normal(sigmas, mus, T):
    nps_lst = [NestedPredictionInterval(None, None) for _ in range(T)]
    for t in range(T):
        upper_func = SymFunction01('mu + sigma*norm.ppf(1-alpha/2)',
                                   'alpha',
                                   {'mu': mus[t],
                                    'sigma': sigmas[t],
                                    'norm':norm})
        lower_func = SymFunction01('mu - sigma*norm.ppf(1-alpha/2)',
                                   'alpha',
                                   {'mu': mus[t],
                                    'sigma': sigmas[t],
                                    'norm': norm})
        nps_lst[t] = NestedPredictionInterval(upper_func, lower_func)
    return nps_lst

def save_betas_normal(raw_csv_path, output_csv_path, value_name, T):
    all_data = pd.read_csv(raw_csv_path, index_col=0)
    all_dates = all_data.index
    all_data = all_data.loc[~all_dates.duplicated(keep='first')]


    for i in range(len(all_data)-1):
        date = all_dates[i]
        true_val = float(all_data.loc[date][value_name])


        preds_str = ['pred_{}'.format(k) for k in range(1, T)]
        ses_str = ['se_{}'.format(k) for k in range(1, T)]
        mus = np.array(all_data.loc[date][preds_str])
        sigmas = np.array(all_data.loc[date][ses_str])

        nps = make_nested_pred_sets_normal(sigmas, mus, 1)[0]
        beta = nps.beta_threshold(true_val)
        all_data.loc[date, 'beta'] = beta
    all_data = all_data.dropna()
    all_data.to_csv(output_csv_path)