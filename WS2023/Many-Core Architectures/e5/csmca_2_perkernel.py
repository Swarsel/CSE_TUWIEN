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
fln = "2_perkernel.cu"
myobj = {'src': open(fln, "r").read(),
         'userargs': ' '.join(sys.argv[2:]),
         'grind': 'none',   # Possible values: none, valgrind, valgrindfull, memcheck, racecheck, synccheck, initcheck
         'profiler': 'none'} # Possible values: none, nvprof

times = []
times2 = []
times3 = []
times4 = []
times5 = []
times6 = []
times7 = []
# ttimes = []
bws = []
its = 7 # it just takes too long otherwise
Ns = [10,100,1000,2000]
for N in Ns:
    print()
    time = []
    time2 = []
    time3 = []
    time4 = []
    time5 = []
    time6 = []
    time7 = []
    for it in range(its):
        myobj['userargs'] = str(N)
        response = requests.post(url, data = myobj)
        add = response.text.split("pre")[8].replace("<","").replace("/","").replace(">","").replace("\n","")
        # print(f"{it}. run time {add[:-1]}")
        print(add)
        out = add.split()
        time.append(float(out[0]))
        time2.append(float(out[1]))
        time3.append(float(out[2]))
        time4.append(float(out[3]))
        time5.append(float(out[4]))
        time6.append(float(out[5]))
        time7.append(float(out[6]))

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
    print(f"time2: {total_time2}, ",end="")
    times2.append(total_time2)

    time3.sort()
    times_kept3 = time3[2:-2]
    total_time3 = 0
    for t in times_kept3:
        total_time3 += t
    total_time3 /= len(times_kept3)
    print(f"time3: {total_time3}, ",end="")
    times3.append(total_time3)

    time4.sort()
    times_kept4 = time4[2:-2]
    total_time4 = 0
    for t in times_kept4:
        total_time4 += t
    total_time4 /= len(times_kept4)
    print(f"time4: {total_time4}, ",end="")
    times4.append(total_time4)

    time5.sort()
    times_kept5 = time5[2:-2]
    total_time5 = 0
    for t in times_kept5:
        total_time5 += t
    total_time5 /= len(times_kept5)
    print(f"time5: {total_time5}, ",end="")
    times5.append(total_time5)

    time6.sort()
    times_kept6 = time6[2:-2]
    total_time6 = 0
    for t in times_kept6:
        total_time6 += t
    total_time6 /= len(times_kept6)
    print(f"time6: {total_time6}, ",end="")
    times6.append(total_time6)

    time7.sort()
    times_kept7 = time7[2:-2]
    total_time7 = 0
    for t in times_kept7:
        total_time7 += t
    total_time7 /= len(times_kept7)
    print(f"time7: {total_time7}, ",end="")
    times7.append(total_time7)
    # ttime = total_time - total_time2
    # ttimes.append(ttime)

    # 8: sizeof(double)
    # bw = 8 * 2 * N / ttime / 1e9;
    # print(f"bw: {bw}, ")
    # bws.append(bw)

with open(f"data/{fln}_rawtimes", "w+") as fil:
    for nt in times:
        fil.write(str(nt))

with open(f"data/{fln}_rawtimes2", "w+") as fil:
    for nt in times2:
        fil.write(str(nt))

with open(f"data/{fln}_rawtimes3", "w+") as fil:
    for nt in times3:
        fil.write(str(nt))

with open(f"data/{fln}_rawtimes4", "w+") as fil:
    for nt in times4:
        fil.write(str(nt))

with open(f"data/{fln}_rawtimes5", "w+") as fil:
    for nt in times5:
        fil.write(str(nt))

with open(f"data/{fln}_rawtimes6", "w+") as fil:
    for nt in times6:
        fil.write(str(nt))

with open(f"data/{fln}_rawtimes7", "w+") as fil:
    for nt in times7:
        fil.write(str(nt))

# with open(f"data/{fln}_ttimes", "w+") as fil:
#     for nt in ttimes:
#         fil.write(str(nt))
# with open(f"data/{fln}_bws", "w+") as fil:
#     for nb in bws:
#         fil.write(str(nb))

plt.loglog(Ns, times, label="vecmat")
plt.loglog(Ns, times2, label="dot1")
plt.loglog(Ns, times3, label="dot2")
plt.loglog(Ns, times4, label="it1")
plt.loglog(Ns, times5, label="it2")
plt.loglog(Ns, times6, label="dot3")
plt.loglog(Ns, times7, label="it3")
plt.title("Runtimes of Conjugate Gradient kernels on RTX3060")
plt.xlabel("$\sqrt{N}$")
plt.ylabel("time [s]")
plt.grid()
plt.legend()
plt.savefig(f"plots/{fln}_timecomp.png")
plt.show()
