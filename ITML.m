function [bregman_div, A] = ITML(y, X, task)

cf = cd('./ITML_code'); setpaths3(); cd(cf)

[bregman_div, A] = MetricLearningAutotune(@ItmlAlg, y, X,[], task);
