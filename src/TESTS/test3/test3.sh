#!/bin/bash

../server.out test3.txt & server_pid=$!
./spawnClients.sh & spawn_pid=$!
sleep 30
kill $spawn_pid
killall client.out
kill -s SIGINT $server_pid  
