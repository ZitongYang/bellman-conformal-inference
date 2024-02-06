import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from utils.dp import DynamicConformal
from utils import save_df, save_pickle, read_exp
from utils import gen_namestr
from visualize import visualize_single_exp
from visualize import gen_plot_data
from dataloader.vlfc import VolatilityData
from dataloader.normfc import ReturnData

import pandas as pd
import numpy as np
from tqdm import tqdm
import time


class ForecastingExperiment:
    def __init__(self, params):
        self.params = params

        if params['task']=='vlfc':
            self.fcdata = VolatilityData(self.params['id'])
        elif params['task']=='rtfc':
            self.fcdata = ReturnData(self.params['id'])
        elif params['task']=='trend':
            self.fcdata = TrendData(self.params['id'])
        else:
            raise NotImplementedError("Dataset not supported")

        self.namestr = gen_namestr(params)

        self.result = None
        self.perfstats = None


    def run(self):
        self.result = pd.DataFrame(columns = ['beta', 'alpha', 'upper',
                                              'lower', 'lambda', 'index',
                                              'true_y'])
    
        # load ogd params if using either aci or bci
        method = self.params['method']
        alpha0 = self.params['alpha0']
        if method=='aci' or method=='bci':
            lambda_max = self.params['lambda_max']
            lambda_min = self.params['lambda_min']
            lbd = self.params['lambda_init']
            gamma = self.params['gamma']
            if method=='bci':
                T = self.params['T']            # look-forward window
                Tp = self.params['Tp']          # look-backward window
    
        ############## start running ##############
        self.fcdata.refresh()
        start_time = time.time()
        logging.info("Experiment {} STARTED".format(self.namestr))
        for t in range(self.fcdata.expectancy()):
        # for t in tqdm(range(self.fcdata.expectancy())): # tqdm
            onedaydata = self.fcdata.next()
          
            # alpha selection
            if  method=='aci' or method=='bci':
                if t>0: # computing the error indicator of k-1
                    errtp1 = float(self.result.loc[t-1]['alpha']>
                                   self.result.loc[t-1]['beta'])
                    # performing the ACI update
                    lbd = lbd - gamma * (alpha0 - errtp1)
                if lbd >= lambda_max:
                    alpha = 0
                elif lbd <= lambda_min:
                    alpha = 1
                else:
                    if method=='bci':
                        if t>Tp: # computing the past mis-coverages
                            rhoTp = np.sum(
                                self.result.loc[t - Tp:t - 1]['alpha'] >
                                self.result.loc[t - Tp:t - 1]['beta'])
                        else:
                            rhoTp = int(alpha0*Tp)
                        dc = DynamicConformal(
                                T, alpha0,
                                [onedaydata['beta_cdf'] for _ in range(T)],
                                [onedaydata['nested_pred_sets'][t].length() 
                                 for t in range(T)],
                                lbd, rhoTp, Tp)
                        dc.dp(bins=200)
                        alpha = dc.optimal_policy[0].eval(0)
                    else:
                        alpha = 1-lbd
            else:
                alpha = alpha0
    
            # post alpha selection
            interval_upper = onedaydata['nested_pred_sets'][0].upper.eval(
                                                                        alpha)
            interval_lower = onedaydata['nested_pred_sets'][0].lower.eval(
                                                                        alpha)
    
            new_row = {'beta': onedaydata['beta'],
                       'alpha': alpha,
                       'upper': interval_upper,
                       'lower': interval_lower,
                       'lambda': lbd if (method=='aci' or method=='bci') else None,
                       'index': onedaydata['index'],
                       'true_y':onedaydata['true_y']}
            self.result = pd.concat([self.result,
                                    pd.DataFrame([new_row])],
                                    ignore_index=True)
        logging.info("Experiment {} ENDED in {}s".format(
            self.namestr,
            time.time()-start_time,))
        self.fcdata.refresh()
        ############## end running ##############
        
        self.result = self.result.set_index('index')


    def save(self):
        assert not (self.result is None)

        # saving result data frame
        save_df(self.namestr, self.result)
        # saving params
        save_pickle(self.namestr, self.params)
        # save result as figures
        self.perfstats = visualize_single_exp(self.params, self.result, returnprefstats=True, showplot=False)

    def load(self):
        self.result, params = read_exp(self.namestr)
        assert params == self.params
        self.perfstats = visualize_single_exp(self.params, self.result, returnprefstats=True, showplot=False)