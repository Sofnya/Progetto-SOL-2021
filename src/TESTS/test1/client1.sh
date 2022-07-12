#!/bin/bash
../client.out -p -t 200 -f test1Address -D miss -W  test1.txt -d read -r test1.txt
../client.out -p -t 200 -f test1Address -c test1.txt