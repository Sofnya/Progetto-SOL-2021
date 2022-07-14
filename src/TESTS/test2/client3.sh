#!/bin/bash
../client.out  -t 200 -p -f test2Address -D miss -W testFiles/hex
../client.out  -t 200 -p -f test2Address -l testFiles/hex -u testFiles/hex -c testFiles/hex
