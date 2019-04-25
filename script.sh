#!/bin/bash -l
 
# Specify the version of MATLAB to be used
module load matlab/2018b

# program name or command and its options and arguments
# matlab -nodisplay -singleCompThread -r test(dset)
matlab -nodisplay -singleCompThread -r "test_pairwise($1), exit"


