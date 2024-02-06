from abc import abstractmethod
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


class ForecastingData:
    """
    Information containing data from a forecasting task
    """
    def __init__(self, task, id):
        self.task = task
        self.id = id
        self.name = '{}-{}'.format(self.task, self.id)

    @abstractmethod
    def next(self):
        """
        :return: {'true_y':,
                  'nested_pred_sets':,
                  'beta_cdf':,
                  'index':}
        """
        pass

    @abstractmethod
    def refresh(self):
        """
        Reset the experiment object so that the .next() starts
        from the begining
        """
        pass

    @abstractmethod
    def expectancy(self):
        """
        Steps until experiment finish
        """
        pass



    def plot_ecc(self, num_alpha_grid=21):
        self.refresh()

        alphas = np.linspace(1e-3, 1, num_alpha_grid)
        miscov_mat = np.empty((0, num_alpha_grid)) 
    
        # Iterate over the dates and record whether true_y 
        # is covered for each alpha in alphas
        expdata = self.next()

        for _ in tqdm(range(self.expectancy())):
            true_y = expdata['true_y']
            nested_pred_sets = expdata['nested_pred_sets']
            miscov_vec= np.ones(num_alpha_grid)
            for alpha_idx, alpha in enumerate(alphas):
                if nested_pred_sets[0].cover(alpha, true_y) != 'under cover':
                    miscov_vec[alpha_idx] = 0
            miscov_vec = miscov_vec.reshape(1, -1)        
            miscov_mat = np.vstack((miscov_mat, miscov_vec))
            
            expdata = self.next()
        self.refresh()

    
        # Compute the coverage for each alpha and plot the results
        coverage_rates = miscov_mat.mean(axis=0)
    
        plt.plot(alphas, coverage_rates, label='ECC')
        plt.plot(alphas, alphas, linestyle='--', label='45-degree line')
        # plt.xlabel('Alpha', fontsize=25)
        plt.ylabel('Mis-coverage Rate', fontsize=25)
        plt.title('Expected Calibration Curve for {}'.format(self.name))
        plt.legend(fontsize=25)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(
            'data/{}/figures/{}-ece.pdf'.format(self.task, self.name),
            bbox_inches='tight', pad_inches=0.01)
        plt.close()