#!/bin/bash
../client.out  -t 200 -p -f test2Address -D miss -W  testFiles/huge/file1 -d read -r testFiles/huge/file1
../client.out -t 200 -p -f test2Address -c testFiles/huge/file1