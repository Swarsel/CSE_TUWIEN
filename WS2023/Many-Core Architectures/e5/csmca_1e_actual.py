#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Wrapper script to send a CUDA source file to the remote exercise environment and retrieve the results.

   Usage: ./csmca.py FILE.cu ARG0 ARG1 ARG2 ...

   where
     FILE.cu is the respective source file with your code
     ARG0 ARG1 ARG2 ... are user-supplied arguments and will be passed to the executable.

   If you have any suggestions for improvements of this script, please contact:

   Author: Karl Rupp <rupp@iue.tuwien.ac.at>
   Course: Computational Science on Many Core Architectures, 360252, TU Wien

   ---

   Copyright 2022, Karl Rupp

   Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

"""

import matplotlib.pyplot as plt
import requests
import sys
# fix default matplotlib in nix trying to use qt when another library using qt is installed
from matplotlib import use as matplotlib_use
matplotlib_use("TkAgg")


#url = 'https://k40.360252.org/2022/ex4/run.php'
url = 'https://rtx3060.360252.org/2023/ex5/run.php'


# Check whether a source file has been passed:
# if len(sys.argv) < 2:
#   print("ERROR: No source file specified")
#   sys.exit()

# Read source file contents:
# try:
#   src_file = open(sys.argv[1], "r")
# except FileNotFoundError:
#   print('ERROR: Source file does not exist!')
#   sys.exit()
# sources = src_file.read()


# Set up JSON object to hold the respective fields, then send to the server and print the returned output (strip HTML tags, don't repeat the source code)
fln = "1e.cu"
myobj = {'src': open(fln, "r").read(),
         'userargs': ' '.join(sys.argv[2:]),
         'grind': 'none',   # Possible values: none, valgrind, valgrindfull, memcheck, racecheck, synccheck, initcheck
         'profiler': 'none'} # Possible values: none, nvprof

times = []
times2 = []
flops = []
ttimes = []
bws = []
its = 14
Ns = [100,1000,10000,100000,300000]
for N in Ns:
    time = []
    time2 = []
    for it in range(its):
        myobj['userargs'] = str(N)
        response = requests.post(url, data = myobj)
        add = response.text.split("pre")[5].replace("<","").replace("/","").replace(">","").replace("\n","")
        # print(f"{it}. run time {add[:-1]}")
        out = add.split()
        time.append(float(out[0]))
        time2.append(float(out[1]))


    print(f"N=: {N}; ", end="")
    #i noticed sporadic high execution times, possibly because many other people are using the machine
    # hence i decided to use the average with some of the highest and lowest times omitted
    time.sort()
    times_kept = time[2:-2]
    total_time = 0
    for t in times_kept:
        total_time += t
    total_time /= len(times_kept)
    print(f"time: {total_time}, ",end="")
    times.append(total_time)

    time2.sort()
    times_kept2 = time2[2:-2]
    total_time2 = 0
    for t in times_kept2:
        total_time2 += t
    total_time2 /= len(times_kept2)
    print(f"time_ref: {total_time2}, ",end="")
    times2.append(total_time2)

    ttime = total_time - total_time2
    ttimes.append(ttime)

    flop = 1000*4096*1024*2/ttime/1e9
    flops.append(flop)

    print(f"flp: {flop}, ")


with open(f"data/{fln}_rawtimes", "w+") as fil:
    for nt in times:
        fil.write(str(nt))
with open(f"data/{fln}_flops", "w+") as fil:
    for nt in flops:
        fil.write(str(nt))

plt.loglog(Ns, flops)
plt.title("Measuring peak GFlOPs/s for RTX3060")
plt.xlabel("N")
plt.ylabel("FlOPs/s")
plt.grid()
plt.savefig(f"plots/{fln}_flpplt.png")
plt.show()
