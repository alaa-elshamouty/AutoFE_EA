# Automated Feature Engineering of Non-categorical Features using Regularized Evolutionary Algorithm for TabPFN

A method for automating the feature engineering process for non-categorical tabular datasets using an evolutionary algorithm and Hyperband Bayesian Optimization to improve the predictive performance of TabPFN.

## Architecture
![Picture1](https://github.com/alaa-elshamouty/AutoFE_EA/assets/47720123/0fa671b0-29d2-4c94-9632-732eea6a0388)
For a given non-categorical dataset, a 70% training and 30\% testing split is performed. The general approach involves passing the training subset to the EA for feature engineering. A set of hyperparameters $\theta$ for the EA is sampled from a defined configuration space (CS) using the BOHB algorithm. The output of the feature engineering process includes the modified training subset $Train'$ and the trajectory $\tau$ of operations leading to these modifications. Inner cross-validation with $k=5$ folds is then performed on $Train'$ to compute the average validation score of the TabPFN model, which serves as the fitness score for the EA. This fitness score is used as feedback for BOHB to sample a new set of hyperparameters and update the EA's population. After several runs, BOHB outputs the best-found set of hyperparameters $\theta^\*$. The EA, using the hyperparameter set $\theta^\*$, is then applied to the initial training subset $Train$. The new best training subset found, denoted as $Train^\*$, is used to train the TabPFN model, and the respective trajectory $\tau^\*$ is applied to the test subset to obtain $Test^\*$. The predictive accuracy of the TabPFN model on $Test^\*$ is the metric used to evaluate the performance after applying automated feature engineering.

## Instructions to build the container from scratch.

For ease of instruction, assume you're working in a folder called *project*. Here are the instructions to build the container:

1.  Install [Singularity](https://docs.sylabs.io/guides/3.0/user-guide/installation.html) on your machine.
2.  Clone the AutoFE repository inside *project*
3.  Move autofe_def.def from *AutoFE_EA* to *project*
5.  With *project* as the root directory, run `singularity build my_container.sif autofe_def.def` to build the container.
6. There are several methods to run in the project:
   1. Run the algorithm to use Hyperband Bayesian Optimization (BOHB) to first optimize the parameters of the evolutionary algorithm and then apply the feature search with the best found set of hyperparameters:
    `singularity exec my_container.sif python3 AutoFE_EA/main.py`
   2. If you previously ran BOHB and the best configuration are saved, then you can run the EA with these configurations: `singularity exec my_container.sif python3 AutoFE_EA/main.py -- evaluate`
   3. If you want to run EA with a set of predefined hyperparameters: `singularity exec my_container.sif python3 AutoFE_EA/main.py --ea_only`. You can change the parameters in the *main.py* file. 

If you have a wandb account, you can activate wandb logging by flagging
`--wandb`
