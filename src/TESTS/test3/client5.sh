#!/bin/bash
i=$((1 + $RANDOM % 10))
target="testFiles/rand/file$i"
../client.out -t 0 -f test3Address -W $target -l $target -r $target -R10 -u $target -c $target
