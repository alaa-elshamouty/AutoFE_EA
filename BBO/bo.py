import json
import os

import numpy as np
from smac.facade.experimental.psmac_facade import PSMAC

from BBO.hpo import BBO
import ConfigSpace as CS

from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_mf_facade import SMAC4MF
from smac.intensification.hyperband import Hyperband
from smac.intensification.successive_halving import SuccessiveHalving
from smac.scenario.scenario import Scenario
from smac.utils.io.output_writer import OutputWriter
# from smac.utils.io.result_merging import ResultMerger
from data.datasets_handling import normalize_data


class BO(BBO):
    def __init__(self, smac_type, runtime, working_dir, dataset, normalizer=None):
        super().__init__(dataset, normalizer)
        self.cs = ConfigurationSpace()
        self.params = []
        self.smac_type = smac_type
        self.runtime = runtime
        self.working_dir = working_dir
        self._setup_initial_config_space()

    def _setup_initial_config_space(self):
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types,
        # we use continuous, integer and categorical parameters.
        normalize_before_ea = CS.CategoricalHyperparameter('normalize', choices=['True', 'False'], default_value='True')
        max_func_evals = CS.UniformIntegerHyperparameter("total_number_of_function_evaluations", 1, 4, default_value=3)
        pop_size = CS.UniformIntegerHyperparameter("population_size", 2, 10, default_value=3)
        fraction_mutation = CS.UniformFloatHyperparameter('fraction_mutation', lower=0., upper=1., default_value=0.7)
        children_per_step = CS.UniformIntegerHyperparameter("children_per_step", 1, 10, default_value=4)
        max_pop_size = CS.UniformIntegerHyperparameter("max_pop_size", 1, 3, default_value=1)
        parent_selection = CS.CategoricalHyperparameter('selection_type',
                                                        choices=[0,1,2],
                                                        default_value=0)

        self.params = [normalize_before_ea, max_func_evals, pop_size, fraction_mutation, children_per_step,
                       max_pop_size, parent_selection]

        self.cs.add_hyperparameters(self.params)

    def _add_configuration(self, config):
        self.params.append(config)
        self.cs.add_hyperparameter(config)

    def _determine_best_hypers(self, config):
        X_train, X_test, y_train, y_test = self.split
        np.random.seed(0)  # fix seed for comparison
        normalize = config['normalize'] == 'True' if 'normalize' in config else True
        if normalize:
            normalizer, X_train = normalize_data(self.dataset, X_train, X_train=True)
            self.normalizer = normalizer
        print('Setting up EA...')
        # setting EA parameters
        self.results['EA_params'] = config

        optimum = self.run_ea(initial_X=X_train, y=y_train, params=config)
        return 1 - optimum.fitness

    def _write_combined_runs(self, scenario, directory):

        """
        Constructs the combined run files for the parallel smac run
        @param scenario: Scenario object
        @param directory: directory where the combined run is written
        """
        pattern = r'run_\d$'
        result_merger = ResultMerger(output_dir=str(directory), rundir_pattern=pattern)
        output_writer = OutputWriter()

        output_writer.save_configspace(self.cs, str(directory / 'configspace.json'), 'json')
        output_writer.save_configspace(self.cs, str(directory / 'configspace.pcs'), 'pcs_new')
        output_writer.write_scenario_file(scenario)
        result_merger.write_runhistory()
        result_merger.write_trajectory()

    def run_bo_parallel(self):
        working_dir = self.working_dir + '/' + self.smac_type + '/' + self.dataset

        # cs.add_hyperparameter(Constant('device', device))
        # SMAC scenario object
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternative to runtime)
                             "wallclock-limit": self.runtime,  # max duration to run the optimization (in seconds)
                             "cs": self.cs,  # configuration space
                             'output-dir': working_dir,  # working directory where intermediate results are stored
                             "deterministic": "true",
                             # "limit_resources": True,  # Uses pynisher to limit memory and runtime
                             # Then you should handle runtime and memory yourself in the TA
                             # If you train the model on a CUDA machine, then you need to disable this option
                             "memory_limit": 8119,  # adapt this to reasonable value for your hardware
                             })

        # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
        max_epochs = 50
        # intensifier parameters (Budget parameters for BOHB)

        n_workers = 5
        intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_epochs, 'eta': 3}

        # To optimize, we pass the function to the SMAC-object

        if self.smac_type == 'BOHB':
            intensifier = Hyperband
        elif self.smac_type == 'BOSH':
            intensifier = SuccessiveHalving
        else:
            raise ValueError('SMAC type can be either BOHB or BOSH')

        smac = PSMAC(
            facade_class=SMAC4MF,
            validate=False,
            n_workers=n_workers,
            shared_model=True,
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=self._determine_best_hypers,
            intensifier=intensifier,
            intensifier_kwargs=intensifier_kwargs,
            # all arguments related to intensifier can be passed like this
            initial_design_kwargs={'n_configs_x_params': 1,  # how many initial configs to sample per parameter
                                   'max_config_fracs': .2})
        # We have to set the output directories manually for psmac
        scenario.output_dir = working_dir
        scenario.output_dir_for_this_run = working_dir
        smac.output_dir = working_dir

        # Start optimization
        incumbent = smac.optimize()
        # combine parallel run and save it into directory
        # self._write_combined_runs(scenario, working_dir)
        # store your optimal configuration to disk
        opt_config = incumbent.get_dictionary()

        with open(working_dir + '/opt_cfg.json', 'w') as f:
            json.dump(opt_config, f)
        return incumbent

    def run_bo(self):
        working_dir = self.working_dir + '/' + self.smac_type + '/' + str(self.dataset)
        if not os.path.exists(working_dir):
            os.mkdir(working_dir)
        # cs.add_hyperparameter(Constant('device', device))
        # SMAC scenario object
        scenario = Scenario(
            {
                "run_obj": "quality",  # we optimize quality (alternative to runtime)
                "wallclock-limit": 3600,  # max duration to run the optimization (in seconds)
                "cs": self.cs,  # configuration space
                "deterministic": True,
                # Uses pynisher to limit memory and runtime
                # Alternatively, you can also disable this.
                # Then you should handle runtime and memory yourself in the TA
                "limit_resources": False,
                "cutoff": 100,  # runtime limit for target algorithm
                "memory_limit": 8119,
                'verbose': 'DEBUG',  # adapt this to reasonable value for your hardware
            }
        )
        # max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
        max_epochs = 50
        # intensifier parameters (Budget parameters for BOHB)
        intensifier_kwargs = {'initial_budget': 5, 'max_budget': max_epochs, 'eta': 3}

        # To optimize, we pass the function to the SMAC-object
        smac = SMAC4MF(
            scenario=scenario,
            rng=np.random.RandomState(42),
            tae_runner=self._determine_best_hypers,
            intensifier_kwargs=intensifier_kwargs,
        )
        tae = smac.get_tae_runner()
        # Start optimization
        try:
            incumbent = smac.optimize()
        finally:
            incumbent = smac.solver.incumbent

        inc_value = tae.run(config=incumbent)[1]
        opt_config = incumbent.get_dictionary()

        with open(working_dir + '/opt_cfg.json', 'w') as f:
            json.dump(opt_config, f)
        return incumbent
