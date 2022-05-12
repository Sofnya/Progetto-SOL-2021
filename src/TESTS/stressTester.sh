#$/bin/bash

for ((i = 0; i < 10000; i++))
do
    ./apiTester.out &
done