#!/bin/bash -l
 
for i in {1..11}
do
   qsub -pe omp 8 -l mem_per_core=16G -o "out$i" script.sh $i
done