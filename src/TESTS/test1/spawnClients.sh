#!/bin/bash

for i in {1..3}
    do
    "./client$i.sh" &
    done
wait