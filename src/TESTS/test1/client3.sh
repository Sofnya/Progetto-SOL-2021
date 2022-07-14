#!/bin/bash
../client.out -p -t 200 -f test1Address -D miss -W testFiles/huge/file1
../client.out -p -t 200 -f test1Address -l testFiles/huge/file1 -u testFiles/huge/file1 -c testFiles/huge/file1
