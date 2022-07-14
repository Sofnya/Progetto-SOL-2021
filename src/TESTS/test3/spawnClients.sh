#!/bin/bash

while true
    do
    num_children=$(pgrep -c -P$$)
    if [ $num_children -lt 10 ];
    then
        i=$((1 + $RANDOM % 5))
        ./client$i.sh &
    fi
    done
wait