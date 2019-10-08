#!/bin/bash

for ((i=0;i<10;i+=1))
do
  # running goal-directed experiments without any augmentations
	python hiro_baseline.py "AntFall" --evaluate --total_steps 10000000 --relative_goals --use_huber --seed $i
done
