#!/bin/bash
../client.out  -t 200 -f test2Address -D miss -W client3.sh
../client.out  -t 200 -f test2Address -l client3.sh -u client3.sh -c client3.sh
