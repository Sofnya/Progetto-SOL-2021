#!/bin/bash

while true
    do
    num_children=$(pgrep -c -P$$)
    for ((j = $num_children; j < 10; j++))
    do
        i=$((1 + $RANDOM % 4))
        ./client$i.sh &
    done
    done
wait