#!/bin/bash
number_optimization_runs=$1

for ((i=0;i<number_optimization_runs;i++))
do
     python -m scoop CTRNN_ReinforcementLearning_CMA-ES.py
done
