#!/bin/bash
../client.out  -t 200 -f test2Address -D miss -W  test2.txt -d read -r test2.txt
../client.out -t 200 -f test2Address -c test2.txt