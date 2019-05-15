#!/bin/bash -l
 
# for i in 1 3 5 6
# do
#     for j in 1 2 3 4 5
#     do
#         qsub -pe omp 16 -l h_rt=168:00:00 -N "out$i$j" -o "out$i$j" -j y script.sh $i $j
#     done
# done

for i in 1 2 3 4 5 6
do  
    for j in 1 2 3 4 5
    do
        qsub -pe omp 28 -l h_rt=168:00:00 -N "out$i$j" -o "out$i$j" -j y script.sh $i $j
    done
done
