#!/bin/bash
rm -rf CMakeFiles 
rm -f CMakeCache.txt
cmake .
make  -j8
