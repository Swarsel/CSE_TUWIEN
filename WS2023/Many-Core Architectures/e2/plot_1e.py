import matplotlib.pyplot as plt
import numpy as np

go = np.array([[0.013499 , 0.006912 , 0.003591 , 0.001871 , 0.001121 , 0.001217 , 0.001019],
[0.006695 , 0.003466 , 0.001862 , 0.001101 , 0.000878 , 0.000956 , 0.000864],
[0.003476 , 0.001871 , 0.001103 , 0.000878 , 0.000832 , 0.000860 , 0.000835],
[0.001874 , 0.001110 , 0.000880 , 0.000833 , 0.000835 , 0.000833 , 0.000830],
[0.001108 , 0.000879 , 0.000833 , 0.000836 , 0.000832 , 0.000831 , 0.000828],
[0.000878 , 0.000833 , 0.000834 , 0.000832 , 0.000830 , 0.000827 , 0.000827],
[0.000834 , 0.000863 , 0.000835 , 0.000830 , 0.000827 , 0.000825 , 0.000825]]
)
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
