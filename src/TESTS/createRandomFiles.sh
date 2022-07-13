#!/bin/bash
for i in $(seq $1)
do
    rand=$(od -N 4 -t uL -An /dev/urandom | tr -d " ")
    size=$(($rand % $2))
    echo "file$i of size $size";
    head -c $size /dev/urandom > "file$i"
done