#!/bin/bash

valgrind --leak-check=full --show-leak-kinds=all ../server.out test1.txt & server_pid=$!
./spawnClients.sh; kill -s SIGHUP $server_pid 