#!/bin/bash
../client.out -p -t 200 -f test1Address -D miss -W client3.sh
../client.out -p -t 200 -f test1Address -l client3.sh -u client3.sh -c client3.sh
