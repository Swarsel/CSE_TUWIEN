#!/bin/bash
echo $(python csmca.py 2_printsol.cu 100 100 | tail -n -3) >> data/2_cgsol.csv
