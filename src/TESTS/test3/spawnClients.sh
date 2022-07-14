#!/bin/bash

while true
    do
    num_children=$(pgrep -c -P$$)
    if [ $num_children -lt 10 ];
    then
        i=$((1 + $RANDOM % 3))
        ./client$i.sh > /dev/null &
    fi
    done
wait