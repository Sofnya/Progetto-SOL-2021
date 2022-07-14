#!/bin/bash
../client.out -t 0 -p -f test3Address -D miss -W  testFiles/huge/file1 -d read -r testFiles/huge/file1
../client.out -t 0 -p -f test3Address -c testFiles/huge/file1