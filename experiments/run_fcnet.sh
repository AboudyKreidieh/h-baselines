#!/bin/bash

for ((i=0;i<10;i+=1))
do
  # running fcnet experiments
	python run_fcnet.py "AntMaze" --evaluate --seed $i --total_steps 10000000 --use_huber
	python run_fcnet.py "AntPush" --evaluate --seed $i --total_steps 10000000 --use_huber
	python run_fcnet.py "AntFall" --evaluate --seed $i --total_steps 10000000 --use_huber
done
