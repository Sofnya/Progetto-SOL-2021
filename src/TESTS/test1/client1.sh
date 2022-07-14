#!/bin/bash
../client.out -p -t 200 -f test1Address -D miss -W  testFiles/hex -d read -r testFiles/hex
../client.out -p -t 200 -f test1Address -c testFiles/hex