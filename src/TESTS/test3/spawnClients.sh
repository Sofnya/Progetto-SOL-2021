#!/bin/bash

while true
    do
    num_children=$(pgrep -c -P$$)
    if [ $num_children -lt 10 ];
    then
        i=$((1 + $RANDOM % 4))
        ./client$i.sh &
    fi
    done
wait