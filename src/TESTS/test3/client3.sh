#!/bin/bash
i=$((1 + $RANDOM % 100))
target="testFiles/smallRand/file$i"
../client.out -t 0 -f test3Address -W $target -l $target -r $target -R10 -u $target
