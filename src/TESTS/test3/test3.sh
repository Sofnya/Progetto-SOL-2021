#!/bin/bash

valgrind ../server.out test3.txt & server_pid=$!
./spawnClients.sh & spawn_pid=$!
sleep 30
kill -s SIGINT $server_pid  
kill $spawn_pid
killall client.out