#!/bin/bash
# createRandomFiles N K creates N random files of a random size from 0 to K-1 inclusive.
for i in $(seq $1)
do
    rand=$(od -N 4 -t uL -An /dev/urandom | tr -d " ")
    size=$(($rand % $2))
    head -c $size /dev/urandom > "file$i"
done