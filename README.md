Peacewise-Linear Bregman Divergence Learning (PBDL) README File
=================================================

1. Prerequisites:

a) Matlab (We used 2019a though it shoud be compatible with other versions)

b) Gurobi 9.0


2. Quick-start pairwise dimilarity

to test the code, we have provided a demo Matlab script "example.m", which runs PBDL
for pairwise similarity comparisions on Iris data-set. Using the learned Bregman divergence,
the code will print out performance metrics for, clustering, K-nn or similarity ranking, 
based on the task you choose. If everything is working correctly, you should first see tuning for
hyperparameters results. Then you should see :

 -.-.-.-.-.-.  Test Performance .-.-.-.-.-.-.- 

 Rand Index = 98.7 +- 1.3 


 Purity = 99.0 +- 1.0 


 K-NN Accuracy = 99.0 +- 1.0 


 Area under the curve = 99.0 +- 0.8 


 Average Precision = 98.3 +- 1.5 


Similarity comparisions are generated as follows: 1000 random pairs from similar class and 1000
random pairs from different classes. The core file which does the optimziation is "PBDL_core"
You can change the method or data-set or other experiment settings as guided in the code.

3. Quick-start regression

To test the code, we have provided a demo Matlab script "example_regression.m", which runs
PBDL for regression with synthetic data. If everything is working properly you should
see a plot of the regression error which reaches around 0.05 with the full 100 point data-set.
The code file which does the optimization is "PBR.m"
You can change the method to "Mahalanobis regression" as guided in the code. you can also change 
the data-set or other experiment settings. 



4. Other files
data folder: data-sets that we used on our paper "Learning Bregman Divergences"
results: regression and pairwise comparision experiment results of our paper
figure: figures used in our paper

5. Help

For any problems please contact us at: ali.siahkamari@gmail.com We'll be happy to 
resolve your issue and improve our code.
We suggest you to read our paper "Learning to Approximate Bregman Divergences".

6. Notice

You can choose different methods in "example.m" by simply uncommenting them.
These methods include, ITML, Kernelized NCA, GMML and Euclidean
