#!/bin/bash
../client.out -t 200 -p -f test2Address -D miss -w testFiles/smallRand,50
../client.out -t 200 -p -f test2Address -d read -R100