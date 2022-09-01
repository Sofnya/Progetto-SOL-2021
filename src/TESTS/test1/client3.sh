#!/bin/bash
valgrind --leak-check=full --show-leak-kinds=all -s ../client.out -p -t 200 -f test1Address -D miss -W testFiles/huge/file1
valgrind --leak-check=full --show-leak-kinds=all -s ../client.out -p -t 200 -f test1Address -l testFiles/huge/file1 -u testFiles/huge/file1 -c testFiles/huge/file1
