from PIL import Image
import numpy as np
image = Image.open('image.png')
image = image.convert("L")
numpydata = np.flip(1.0 - (np.asarray(image)/255), axis = 0)
numpydata = 2*numpydata - 1

from matplotlib import pyplot as plt
x = np.linspace(0,1,numpydata.shape[0])
X,Y = np.meshgrid(x,x)


plt.pcolor(X,Y,numpydata, cmap = 'Greys') 
plt.colorbar()
plt.savefig(f"./mickey.pdf", format='pdf')
