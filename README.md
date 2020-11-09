# Numerical Exploration of Training Loss Level-Sets in Deep Neural Networks

This directory contains the code for the experiments reported in our working paper titled `Numerical Exploration of Training Loss Level-Sets in Deep Neural Networks`. There are five experiments with abbreviated labels `MNIST_FF`, `MNIST_CONV`, `CIFAR10_CONV`, `IRIS_FF` and `MPG_FF`. The code uses Python 3.7.x.

## What is included?
Each experiment has a corresponding `.py` file that contains the code for our method. For each experiment, we also have an `L2_` pre-fixed `.py` file that includes the code for baseline version of the experiment with weight decay. Furthermore, there are two Jupyter notebooks per experiment: one to visualize the outcome of experiment runs with our method and the other to compute various statistics related to baseline version of the experiment. Lastly, the code for visualizing the whole level-set traversal for `MNIST_FF` is included in `PCA_MNIST_FF.ipynb`.

The various parameters for all experiments can be altered in the files `Configs.py` or `Configs_L2.py`.

## How to set up the runtime environment?
The Python packages required to run the code are given in the included requirements.txt file. Note that we use `%matplotlib widget` [magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) in the notebooks to generate interactive plots using [ipympl](https://github.com/matplotlib/ipympl) (also included in requirements.txt). But you can replace the magic and remove `ipympl` requirement at your discretion.

## What to run?
An example sequence of actions required to reproduce our reported results for, say `MNIST_FF` experiments is as follows:
1. Run the `MNIST_FF.py` file to get the results using our predictor/corrector method.
2. Visualize the results using `Viz_MNIST_FF.ipynb` (similar to Figure 2 in the paper).
3. Run the `L2_MNIST_FF.py` to get the baseline (training with weight decay in the loss) results for the same experiments.
4. Analyze the baseline results using `Viz_L2_MNIST_FF.ipynb` (Table 1 entries in the paper).

Optional: Run `PCA_MNIST_FF.ipynb` to visualize whole level-set traversal (Figure 7 in the paper). Note this particular notebook is included for `MNIST_FF` only, but similar visualiztions for other experiments can be produced by changing only a few lines.


### Notes
Since we reported aggregated statistics for multiple, randomly-initialized runs of each experiment, the effect of setting random seeds in any part of the code is highly-mitigated. But a seed is set explicitly given where doing so is deemed appropriate. For example, we include the seed for Principal Component Analysis calculations in the `PCA_MNIST_FF.ipynb` notebook.
