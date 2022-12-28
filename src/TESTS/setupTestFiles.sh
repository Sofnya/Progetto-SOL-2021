#!/bin/bash
mkdir testFiles

mkdir testFiles/huge
cd testFiles/huge
../../createRandomFiles.sh 1 1000000
cd ../

mkdir smallRand
cd smallRand
../../createRandomFiles.sh 100 1000
cd ../
for i in $(seq 10)
do
    target="rand$i"
    mkdir $target
    cd $target
    ../../createRandomFiles.sh 10 100000
    cd ../
done
xxd -l 10000 -c 1000 -p < /dev/urandom > hex
