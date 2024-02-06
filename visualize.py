import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from utils import read_exp, gen_namestr, \
                  trim, compute_local_avg, compute_local_median, \
                  compute_avg_no_inf
from dataloader.vlfc import COMPANIES

PLOT_PARAM = {'xform': {'vlfc': lambda val: np.sqrt(val),
                        'rtfc': lambda val: val},
              'xstick_skip': {'vlfc': 500,
                              'rtfc': 300},
              'ma_window': {'vlfc': 250,
                            'rtfc': 250},
              'color': {'bci': 'tomato',
                        'fixed': 'silver',
                        'aci': 'navy'},
              'unit': {'vlfc': 'Volatility (%)',
                       'rtfc': 'Log Return'},
              'task_name':{'vlfc': 'Volatility forecasting for ',
                           'rtfc': 'Return forecasting for '},
                'method_name': {'aci': 'ACI',
                                'bci': 'BCI',
                                'fixed': 'Fixed'},
             }


def gen_plot_data(result_df, ma_window, xstick_skip, xform):
    err_ind = np.array(result_df['alpha'] > result_df['beta'])
    upper = xform(result_df['upper'])
    lower = xform(result_df['lower'])

    # running average
    miscovrate = compute_local_avg(err_ind, ma_window) # miscoverage rate
    length = compute_local_avg(upper-lower, ma_window) # length


    # date indices
    date_indices = trim(result_df.index, ma_window)
    date_indices = [date_index.strftime(
                            "%Y-%m" if xstick_skip<100 else "%Y")
                    for date_index in date_indices]
    indices = range(0, len(date_indices), xstick_skip)



    # other variables
    upper = trim(upper,
                 ma_window)
    lower = trim(lower,
                 ma_window)
    truey = trim(xform(np.array(result_df['true_y'])),
                ma_window)
    alpha = trim(np.array(result_df['alpha']),
                 ma_window)
    beta = trim(np.array(result_df['beta']),
                 ma_window)
    lbd   = trim(np.array(result_df['lambda']),
                 ma_window)

    # performance metrics
    perfstats = {
        'avg_miscov':np.mean(err_ind),
        'med_length':np.median(upper-lower),
        'avg_length':compute_avg_no_inf(upper-lower),
        'smoothness':np.mean(np.abs(np.diff(alpha))),
        'excursion': np.std(miscovrate[int(0.1*len(miscovrate)):]),
        'percent_inf_len':np.sum(np.isinf(upper-lower))/len(result_df),
    }

    return {'miscovrate': miscovrate,
            'indices': indices,
            'date_indices': date_indices,
            'upper': upper,
            'lower': lower,
            'true_y': truey,
            'alpha': alpha,
            'beta': beta,
            'lbd': lbd,
            'length': length,
            'perfstats': perfstats}


def visualize_single_exp(exp_params, result_df, returnprefstats=False, savefig=False, showplot=True):
    namestr = gen_namestr(exp_params)
    plotDt = gen_plot_data(result_df,
                           PLOT_PARAM['ma_window'][exp_params['task']],
                           PLOT_PARAM['xstick_skip'][exp_params['task']],
                           PLOT_PARAM['xform'][exp_params['task']])
    # figure setup
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # plotting top miscoverage
    axs[0].plot(100*plotDt['miscovrate'], '-',
            color=PLOT_PARAM['color'][exp_params['method']],
            label=exp_params['method'], linewidth=3)
    axs[0].axhline(y=100*exp_params['alpha0'],
                color='black', linestyle='--', linewidth=1)
    axs[0].set_ylabel('Mis-coverage rate (%)', fontsize=15)
    # axs[0].set_ylim([0, 0.2])
    axs[0].legend(loc='upper right', prop={'size': 12})
    axs[0].set_xticks(plotDt['indices'])
    axs[0].set_xticklabels([plotDt['date_indices'][i] for i in plotDt['indices']],
                        rotation=30)


    # plotting bottom real data visualization
    fill_x = range(len(plotDt['true_y']))
    axs[1].fill_between(range(len(plotDt['true_y'])),
                        plotDt['lower'],
                        plotDt['upper'],
                        color=PLOT_PARAM['color'][exp_params['method']],
                        label=exp_params['method'], alpha=0.6)
    axs[1].plot(plotDt['true_y'], '.',
                color='black', markersize=3)
    axs[1].set_ylabel(PLOT_PARAM['unit'][exp_params['task']], fontsize=15)
    axs[1].legend(loc='upper right', prop={'size': 12})
    axs[1].set_xticks(plotDt['indices'])
    axs[1].set_xticklabels([plotDt['date_indices'][i] for i in plotDt['indices']],
                        rotation=30)

    # Time series of plotDt['alpha'] on the right
    axs[2].plot(plotDt['alpha'], '-', label='Selected nominal coverage',
                color=PLOT_PARAM['color'][exp_params['method']],
                markersize=1)
    axs[2].set_ylabel('Prediction set index', fontsize=15)
    axs[2].legend(loc='upper right', prop={'size': 12})
    axs[2].set_xticks(plotDt['indices'])
    axs[2].set_xticklabels([plotDt['date_indices'][i] for i in plotDt['indices']],
                        rotation=30)

    # making title
    title_str = ('{}-{}-{}: Portion inf.: {}%, '
                'Avg. miscov.: {}%,'
                'Avg. len. (-inf.): {}, '
                'Excursion: {}%').format(
                exp_params['task'],
                exp_params['id'],
                exp_params['method'],
                np.around(100*plotDt['perfstats']['percent_inf_len'],decimals=2),
                np.around(100*plotDt['perfstats']['avg_miscov'],decimals=2),
                np.around(plotDt['perfstats']['avg_length'],decimals=2),
                np.around(100*plotDt['perfstats']['excursion'], decimals=2))

    fig.suptitle(title_str, fontsize=15, y=1.0005)
    plt.tight_layout()
    if not showplot:
        plt.close()

    if savefig:
        plt.savefig('result/figures/single-{}.pdf'.format(namestr))
    if returnprefstats:
        return plotDt['perfstats']

def visualize_three_exp(params1, df1, params2, df2, params3, df3, short_title=False, savefig=False):
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
    params_lst = [params1, params2, params3]
    namestr_lst = [gen_namestr(params1), gen_namestr(params2), gen_namestr(params3)]
    df_lst = [df1, df2, df3]
    task = params1['task']
    assert task == params2['task'] == params3['task']

    title_lines = []
    for i in range(3):
        plotDt = gen_plot_data(df_lst[i],
                               PLOT_PARAM['ma_window'][task],
                               PLOT_PARAM['xstick_skip'][task],
                               PLOT_PARAM['xform'][task])
        title_lines.append('{}-{}-{}: Portion inf.: {}%, '
                           'Avg. miscov.: {}%,'
                           'Avg. len. (-inf.): {},'
                           'Excursion: {}'.format(
            task, params_lst[i]['id'], params_lst[i]['method'],
            np.around(100*plotDt['perfstats']['percent_inf_len'], decimals=2),
            np.around(100*plotDt['perfstats']['avg_miscov'],decimals=2),
            np.around(plotDt['perfstats']['avg_length'], decimals=2),
            np.around(100*plotDt['perfstats']['excursion'], decimals=2)))

        # Panel 0: Moving average of miscoverage rate
        axs[0].plot(100*plotDt['miscovrate'], '-',
                    color=PLOT_PARAM['color'][params_lst[i]['method']],
                    label=PLOT_PARAM['method_name'][params_lst[i]['method']], linewidth=3)

        # Panel 1: Moving average of interval length
        axs[1].plot(plotDt['length'], '-',
                    color=PLOT_PARAM['color'][params_lst[i]['method']],
                    label=PLOT_PARAM['method_name'][params_lst[i]['method']], linewidth=3)
    
    # Set labels and titles for each panel
    axs[0].set_ylabel('Mis-coverage rate (%)', fontsize=25)
    axs[0].axhline(y=100*params_lst[0]['alpha0'],
                   color='black', linestyle='--', linewidth=1)
    axs[0].legend(loc='upper right', prop={'size': 25})
    axs[0].set_xticks(plotDt['indices'])
    axs[0].set_xticklabels([plotDt['date_indices'][i] for i in plotDt['indices']],
                           rotation=30, fontsize=20)
        
    axs[1].set_ylabel('Interval Length', fontsize=25)
    axs[1].legend(loc='upper right', prop={'size': 25})
    axs[1].set_xticks(plotDt['indices'])
    axs[1].set_xticklabels([plotDt['date_indices'][i] for i in plotDt['indices']],
                           rotation=30, fontsize=20)

    if short_title:
        fig.suptitle(PLOT_PARAM['task_name'][task] + params1['id'], fontsize=35, y=1.0005)
    else:
        fig.suptitle('\n'.join(title_lines), fontsize=10, y=1.0005)

    for ax in range(2):
        for label in axs[ax].get_yticklabels():
            label.set_fontsize(20)

    plt.tight_layout()
    if savefig:
        plt.savefig('{}/result/figures/three-{}.pdf'.format('_'.join(namestr_lst)))