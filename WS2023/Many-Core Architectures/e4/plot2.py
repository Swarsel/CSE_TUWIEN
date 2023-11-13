import matplotlib.pyplot as plt
import numpy as np
# own kernel data
aa=[4.07e-05,5.56e-05,0.0002437,0.0021272]
bb=[7.32e-05,0.000106,0.0004825,0.0042479]
cc=[0.000107,0.000152,0.0007203,0.0063723]
dd=[0.0001402,0.0002075,0.0009587,0.0084708]


N= [10000,100000,1000000,10000000]

#cublas data
a=[0.0001437,0.0001478,0.0005317,0.003879]
b=[0.0002832,0.0002889,0.0010534,0.007751]
c=[0.0004227,0.0004307,0.0015822,0.0116129]
d=[0.0005624,0.000573,0.0020597,0.0154853]

plt.title("Comparison of mdot kernels")
plt.xlabel("N")
plt.ylabel("Time [s]")
plt.loglog(N, aa,label="Own kernel, $k=8$")
plt.loglog(N, bb,label="Own kernel, $k=16$")
plt.loglog(N, cc,label="Own kernel, $k=24$")
plt.loglog(N, dd,label="Own kernel, $k=32$")
plt.loglog(N, a,linestyle="--", label="CUBLAS, $k=8$")
plt.loglog(N, b,linestyle="--",label="CUBLAS, $k=16$")
plt.loglog(N, c,linestyle="--",label="CUBLAS, $k=24$")
plt.loglog(N, d,linestyle="--",label="CUBLAS, $k=32$")
plt.grid()
plt.legend()
plt.savefig("comp2.jpg")

plt.show()
