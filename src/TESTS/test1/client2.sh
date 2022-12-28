#!/bin/bash
../client.out -p -t 200 -f test1Address -D miss -w testFiles/smallRand,10
../client.out -p -t 200 -f test1Address -d read -R100