#!/bin/bash

cd "./TQC/"
./run_test.sh &

sleep 3

cd "../Decision Transformer/"
./run_test.sh &

wait

echo "All tests have finished running."

