#!/bin/bash

for ((i=0;i<10;i+=1))
do
  # running goal-conditioned experiments without any augmentations
	python run_hrl.py "AntMaze" --evaluate --total_steps 10000000 --relative_goals --use_huber --seed $i
done
