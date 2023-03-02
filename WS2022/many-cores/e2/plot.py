import matplotlib.pyplot as plt
import numpy as np

'''x = [10,100,1000,10000,100000,1000000,10000000]
x_vecadd = [100,300,1000,10000,100000,1000000,3000000]
gpugpu = [0.000070,0.000077,0.000080,0.000131,0.000525,0.004015,0.011726]
gpucpu = [0.000086,0.000090,0.000100,0.000170,0.001030,0.007627,0.021745]
vecadd = [0.000026,0.000019,0.000019,0.000020,0.000031,0.000162,0.000424]

plt.title("Comparison of vector addition for differing N")
#plt.title("Comparison of dot products")
plt.ylabel("time [s]")
plt.xlabel("N")
plt.semilogx(x_vecadd, vecadd)
#plt.semilogx(x, gpugpu, label="2 kernels")
#plt.semilogx(x, gpucpu, label="1 kernel + cpu")
#plt.legend()
plt.show()'''
'''
go = np.array([[0.000012 , 0.000010 , 0.000010 , 0.000010 , 0.000012 , 0.000013 , 0.000016 ],
 [0.000010 , 0.000010 , 0.000010 , 0.000010 , 0.000012 , 0.000013 , 0.000016  ],
 [0.000010 , 0.000011 , 0.000010 , 0.000011 , 0.000012 , 0.000014 , 0.000017  ],
 [0.000010 , 0.000010 , 0.000010 , 0.000011 , 0.000012 , 0.000015 , 0.000019  ],
 [0.000010 , 0.000011 , 0.000011 , 0.000013 , 0.000015 , 0.000019 , 0.000027  ],
 [0.000010 , 0.000011 , 0.000013 , 0.000015 , 0.000019 , 0.000028 , 0.000046  ],
 [0.000011 , 0.000013 , 0.000015 , 0.000020 , 0.000029 , 0.000047 , 0.000083]])
'''
go = np.array([[0.033051 , 0.016575 , 0.008385 , 0.004257 , 0.004152 , 0.003140 , 0.002652] ,
[0.017971 , 0.009011 , 0.004612 , 0.002345 , 0.002289 , 0.001741 , 0.001468] ,
[0.009010 , 0.004610 , 0.002343 , 0.001186 , 0.001326 , 0.001004 , 0.000932] ,
[0.004616 , 0.002347 , 0.001186 , 0.000813 , 0.000927 , 0.000854 , 0.000718],
[0.002351 , 0.001188 , 0.000819 , 0.000935 , 0.000851 , 0.000717 , 0.000746] ,
[0.001205 , 0.000821 , 0.000937 , 0.000851 , 0.000718 , 0.000745 , 0.000701] ,
[0.000828 , 0.000940 , 0.000863 , 0.000721 , 0.000744 , 0.000696 , 0.000724]])
xx = [16,32,64,128,256,512,1024]
yy = [16,32,64,128,256,512,1024]

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
        text = ax.text(j, i, "",
                       ha="center", va="center", color="w")

ax.set_title("Comparison of vector addition for different grid-/blocksizes; $N=10^7$")
fig.tight_layout()
plt.colorbar(im)
plt.show()