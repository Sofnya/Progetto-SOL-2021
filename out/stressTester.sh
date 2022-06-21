#$/bin/bash

for ((i = 0; i < 500; i++))
do
    ./apiTester.out &
done
