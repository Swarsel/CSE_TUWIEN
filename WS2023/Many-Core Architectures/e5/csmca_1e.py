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
import numpy as np

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

# mat4 = []
times = []
# times2 = []
# ttimes = []
persecs = []
totals =[]
# bws = []
it1d = 1000
its = 10 # this takes too long otherwise
sizes = [2 ** p for p in range(8,11)]
# go = np.array([sizes[0:1], sizes[1:2], sizes[2:3]])
for bl in sizes:
    for th in sizes:

         time = []
         worked = 100000000 * 2
         for it in range(its):
             myobj['userargs'] = f"{bl} {th}"
             response = requests.post(url, data = myobj)
             add = response.text.split("pre")[5].replace("<","").replace("/","").replace(">","").replace("\n","")
             # print(f"{it}. run time {add[:-1]}")
             out = add.split()
             #print(out[0])
             time.append(float(out[0]))

         print(f"bl=: {bl}; ", end="")
         print(f"th=: {th}; ", end="")
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

         persec = worked / total_time / 1e9
         totals.append((bl, th, persec))

         # 8: sizeof(double)
         print(f"FlOPs: {persec}, ")
         persecs.append(persec)

with open(f"data/{fln}_rawtimes", "w+") as fil:
    for nt in times:
        fil.write(str(nt))
with open(f"data/{fln}_flopss", "w+") as fil:
    for nt in persecs:
        fil.write(str(nt))
with open(f"data/{fln}_totals", "w+") as fil:
    for nt in totals:
        fil.write(str(nt))
# with open(f"data/{fln}_bws", "w+") as fil:
#     for nb in bws:
#         fil.write(str(nb))


elem = len(sizes)
go = np.array([persecs[0:elem], persecs[elem:elem*2], persecs[elem*2:elem*3]])

xx = [256,512,1024]
yy = [256,512,1024]

fig, ax = plt.subplots()
im = ax.imshow(go)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(yy)), labels=yy)
ax.set_yticks(np.arange(len(xx)), labels=xx)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(xx)):
    for j in range(len(yy)):
        text = ax.text(j, i, np.format_float_scientific(go[i][j],precision=5),
                       ha="center", va="center", color="w")

ax.set_title("Searching for max GFlOPs/s")
fig.tight_layout()
plt.colorbar(im)
plt.savefig(f"plots/{fln}_aaplt.png")
plt.show()
