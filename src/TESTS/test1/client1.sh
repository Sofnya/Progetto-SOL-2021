#!/bin/bash
valgrind --leak-check=full --show-leak-kinds=all -s ../client.out -p -t 200 -f test1Address -D miss -W  testFiles/hex -d read -r testFiles/hex
valgrind --leak-check=full --show-leak-kinds=all -s ../client.out -p -t 200 -f test1Address -c testFiles/hex