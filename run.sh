#!/bin/bash
echo "Starting Flower Server..."
python server.py &

sleep 5

echo "Starting Client 1..."
DATA_DIR=./hospital_1/data python client.py &

echo "Starting Client 2..."
DATA_DIR=./hospital_2/data python client.py &
