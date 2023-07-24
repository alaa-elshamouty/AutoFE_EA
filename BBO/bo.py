import json
import os

import ConfigSpace as CS
import numpy as np
from smac.configspace import ConfigurationSpace
from smac.facade.smac_mf_facade import SMAC4MF
from smac.scenario.scenario import Scenario

from BBO.hpo import BBO


class BO(BBO):
    def __init__(self, dataset, smac_type='BOHB', runtime=21600, working_dir='results_bo'):
        super().__init__(dataset)
        self.cs = ConfigurationSpace()
        self.params = []
        self.smac_type = smac_type
        self.runtime = runtime
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        self.working_dir = working_dir
        self._setup_initial_config_space()

    def _setup_initial_config_space(self):
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types,
        # we use continuous, integer and categorical parameters.
        max_func_evals = CS.CategoricalHyperparameter('total_number_of_function_evaluations',
                                                      choices=[100, 500, 1000, 2000], default_value=100)
        pop_size = CS.CategoricalHyperparameter('population_size', choices=[3, 10, 50], default_value=3)
        fraction_mutation = CS.UniformFloatHyperparameter('fraction_mutation', lower=0., upper=1., default_value=0.5)
        children_per_step = CS.CategoricalHyperparameter('children_per_step', choices=[1, 3, 10], default_value=1)
        max_pop_size = CS.CategoricalHyperparameter('max_pop_size', choices=[0, 10, 50, 100], default_value=0)
        parent_selection = CS.CategoricalHyperparameter('selection_type',
                                                        choices=[0, 1, 2],
                                                        default_value=0)
        regularizer = CS.UniformFloatHyperparameter("regularizer", 0, 1, default_value=0.25)

        self.params = [max_func_evals, pop_size, fraction_mutation, children_per_step,
                       max_pop_size, parent_selection, regularizer]

        self.cs.add_hyperparameters(self.params)

    def _add_configuration(self, config):
        self.params.append(config)
        self.cs.add_hyperparameter(config)

    def _determine_best_hypers(self, config):
        X_train, X_test, y_train, y_test = self.split
        np.random.seed(0)  # fix seed for comparison
        print('Setting up EA...')
        # setting EA parameters
        self.results['EA_params'] = config

        optimum = self.run_ea(job_name='searching_config', X=X_train, y=y_train, X_test=X_test, y_test=y_test,
                              params=config)
        return 1 - optimum.fitness

    def run_bo(self):
        working_dir = os.path.join(self.working_dir, str(self.dataset))
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        # SMAC scenario object
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternative to runtime)
                "wallclock-limit": self.runtime,  # max duration to run the optimization (in seconds)
                "cs": self.cs,  # configuration space
                "deterministic": True,
                # Uses pynisher to limit memory and runtime
                # Alternatively, you can also disable this.
                # Then you should handle runtime and memory yourself in the TA
                "limit_resources": False,
                "cutoff": 3 * 60 * 60,  # runtime limit for target algorithm
                "memory_limit": 8119,
                "abort_on_first_run_crash": False
            }
        )
        # max budget for hyperband
        max_epochs = 1000
        # intensifier parameters (Budget parameters for BOHB)
        intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_epochs, 'eta': 3}

        # To optimize, we pass the function to the SMAC-object
        smac = SMAC4MF(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=self._determine_best_hypers,
            intensifier_kwargs=intensifier_kwargs,
        )
        runs_working_dir = os.path.join(working_dir, 'runs')
        if not os.path.exists(runs_working_dir):
            os.mkdir(runs_working_dir)
        smac.output_dir = runs_working_dir
        # Start optimization
        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        opt_config = incumbent.get_dictionary()

        with open(working_dir + '/opt_cfg.json', 'w') as f:
            json.dump(opt_config, f)
        return opt_config
