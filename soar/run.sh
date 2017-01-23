#!/bin/bash

## 10 points, 20 basis
#for i in `seq 0 9`; do
#    mpirun -np 4 ./soar $((40+$i*80)) 20
#    wait
#done

## 20 points, 20 basis
for i in `seq 0 9`; do
    mpirun -np 4 ./soar $((40+$i*80)) 20
    wait
done
