A method for automating the feature engineering process for non-categorical tabular datasets using an evolutionary algorithm and Hyperband Bayesian Optimization to improve the predictive performance of TabPFN.


## Instructions to build the container from scratch.

For ease of instruction, assume you're working in a folder called *project*. Here are the instructions to build the container:

1.  Install [Singularity](https://singularity.lbl.gov/install-linux) on your machine.
2.  Clone the AutoFE repository inside *project*
3.  Move autofe_def.def from AutoFE to *project*
5.  With project as the root directory, run singularity build my_container.sif autofe_def.def to build the container.
6. There are several methods to run in the project:
   1. Run the project to use Hyperband Bayesian Optimization (BOHB)to first optimize the parameters of the evolutionary algorithm and then apply the feature search with the best found set of hyperparameters:
    `singularity run my_container.sif python3.8 AutoFE_EA/main.py`
   2. If you previously ran BOHP and the best configuration are saved, then you can run teh EA with these configurations: `singularity run my_container.sif python3.8 AutoFE_EA/main.py -- evaluate`
   3. If you want to run EA with predefined hyperparameters: `singularity run my_container.sif python3.8 AutoFE_EA/main.py --ea_only`. You can change the parameters in the main.py file. 
      