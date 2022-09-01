#!/bin/bash
valgrind --leak-check=full --show-leak-kinds=all -s ../client.out -p -t 200 -f test1Address -D miss -w testFiles,10
valgrind --leak-check=full --show-leak-kinds=all -s ../client.out -p -t 200 -f test1Address -d read -R100