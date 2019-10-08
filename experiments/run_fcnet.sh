#!/bin/bash

for ((i=0;i<10;i+=1))
do
  # running fcnet experiments
	python fcnet_baseline.py "AntMaze" --evaluate --seed $i --total_steps 10000000 --use_huber
	python fcnet_baseline.py "AntPush" --evaluate --seed $i --total_steps 10000000 --use_huber
	python fcnet_baseline.py "AntFall" --evaluate --seed $i --total_steps 10000000 --use_huber
done
