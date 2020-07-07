Peacewise-Linear Bregman Divergence Learning (PBDL) README File
=================================================

1. Prerequisites:

a) Matlab (We used 2019a though it shoud be compatible with other versions)
b) For faster computation time, install Gurobi optimization and its matlab interface from "gurobi.com" 

2. Quick-start regression

To test the code, we have provided a demo Matlab script "test_regression.m", which runs
PBDL for regression with synthetic data. If everything is working properly you should
see a plot of the regression error which reaches around 0.05 with the full 100 point data-set.
The code file which does the optimization is "PBR.m"
You can change the method to "Mahalanobis regression" as guided in the code. you can also change 
the data-set or other experiment settings. 

3. Quick-start pairwise dimilarity

to test the code, we have provided a demo Matlab script "test_pairwise.m", which runs PBDL
for pairwise similarity comparisions on Iris data-set. Using the learned Bregman divergence,
the code will print out performance metrics for, clustering, K-nn or similarity ranking, 
based on the task you choose. If everything is working correctly, you should first see tuning for
hyperparameters results. Then you should see :

Rand Index = 97.9  -/+  0.0              (95%  approximate confidence interval)


Similarity comparisions are generated as follows: 500 random pairs from similar class and 500
random pairs from different classes. The core file which does the optimziation is "PBDLL1"
You can change the method or data-set or other experiment settings as guided in the code.

4. Other files
data folder: data-sets that we used on our paper "Learning Bregman Divergences"
results: regression and pairwise comparision experiment results of our paper
figure: figures used in our paper

5. Help

For any problems please contact us at: siaa@bu.edu We'll be happy to 
resolve your issue and improve our code.
We suggest you to read our paper [Learning to Approximate Bregman Divergences](https://arxiv.org/pdf/1905.11545.pdf).

6. Notice

You can choose different methods in "test_pairwise.m" by simply uncommenting them.
These methods include, ITML, Kernelized LMNN and Kernelized NCA.
