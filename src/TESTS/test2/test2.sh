#!/bin/bash

../server.out test2.txt & server_pid=$!
./spawnClients.sh; kill -s SIGHUP $server_pid 