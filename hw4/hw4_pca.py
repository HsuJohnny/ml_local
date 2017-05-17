"""
top 9 eigenface need to be modify
"""
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc

# imahes is a list of [ [] ]
# for example, [ [1,2,3] , [1,2,3] ]
images = [] 
o_pics = []
for i in range(10):
	for j in range(10):
		image = misc.imread("faceExpressionDatabase/%s0%s.bmp" %(chr(65+i),str(j)))
		image = image.reshape(4096)
		o_pics.append(image)
		image = image.reshape(64*64)
		images.append(image)

submean_images = []
for image in images:
	i_mean = np.mean(image)
	image = image - i_mean
	submean_images.append(image)
submean_images = np.array(submean_images)
images = np.array(images)
"""
plt_face = images.reshape(100,4096)
avg_face = np.mean(plt_face, axis=0).reshape(64,64)
plt.imshow(avg_face, interpolation='nearest', cmap='gray')
plt.show()
"""
print ("calculating SVD...")
U, s, V = np.linalg.svd(submean_images)
eigenfaces = V
top9_eigenfaces = []
for i in range(9):
	top9_eigenfaces.append(eigenfaces[i])


# project to top 5 eigenface and recovery
"""
n_components = 60
top5_U = U[:,0:n_components]
top5_s = np.diag(s[0:n_components])
top5_V = V[0:n_components,:]
top5_face = np.dot(np.dot(top5_U, top5_s), top5_V)
draw_top5 = []
for item in top5_face:
	item = item.reshape(64,64)
	draw_top5.append(item)
total_err = 0
for pic, o_pic in zip(top5_face, o_pics):
	pic_mean = np.mean(o_pic)
	pic = pic + pic_mean
	pic_err = np.sum(np.power((o_pic - pic), 2))
	total_err += pic_err
total_err /= 409600
total_err = np.power(total_err, 0.5)/256
print (total_err)


#plt.imshow(draw_top9[0], interpolation='nearest', cmap='gray')
width = 64*10
height = 64*10
output = np.zeros((width, height))
i = 0
for y in range(10):
	for x in range(10):
		output[64*x:64*(x+1),64*y:64*(y+1)] = draw_top5[i]
		i = i + 1
plt.imshow(output, interpolation='nearest', cmap='gray')
plt.show()


# draw original 100 face
weight = 64*10
height = 64*10
original = np.zeros((weight, height))
draw_original = []
for item in images:
	item = item.reshape(64,64)
	draw_original.append(item)
i = 0
for y in range(10):
	for x in range(10):
		original[64*x:64*(x+1), 64*y:64*(y+1)] = draw_original[i]
		i = i + 1
plt.imshow(original, interpolation='nearest', cmap='gray')
plt.show()
"""

# plot top 9 face
draw_top9 = []
for item in top9_eigenfaces:
	item = item.reshape(64,64)
	draw_top9.append(item)
#plt.imshow(draw_top9[0], interpolation='nearest', cmap='gray')
width = 64*3
height = 64*3
output = np.zeros((width, height))
i = 0
for y in range(3):
	for x in range(3):
		output[64*x:64*(x+1),64*y:64*(y+1)] = draw_top9[i]
		i = i + 1
plt.imshow(output, interpolation='nearest', cmap='gray')
plt.show()



