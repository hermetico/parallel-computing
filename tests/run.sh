#!/bin/bash
# 4kB ... 256MB
LENGTH=1024
for i in `seq 1 14`;
do
    ./read-vector.exe $LENGTH > $LENGTH.data
    let LENGTH=LENGTH*2
done
