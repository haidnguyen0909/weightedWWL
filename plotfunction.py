import matplotlib.pyplot as plt
import numpy as np

def hinge1(x, alpha1):
	y = []

	for i in range(len(x)):
		if x[i] > alpha1:
			y.append(0)
		else:
			y.append(alpha1-x[i])
	return np.array(y)

def hinge2(x, alpha2):
	y=[]
	for i in range(len(x)):
		if x[i] < alpha2:
			y.append(0.0)
		else:
			y.append(x[i]-alpha2)
	return np.array(y)
def smoothhinge2(x, alpha2, width):
	y=[]
	for i in range(len(x)):
		if x[i] < alpha2:
			y.append(0)
		if x[i]>alpha2+width:
			y.append(x[i]-alpha2-0.5 * width)
		if x[i] <= alpha2+width and x[i] >= alpha2:
			y.append(0.5/width*(x[i]-alpha2)*(x[i]-alpha2))
	return np.array(y)


def smoothhinge1(x, alpha1, width):
	y=[]
	for i in range(len(x)):
		print(i)
		if x[i] > alpha1:
			y.append(0)
		if x[i] < alpha1 - width:
			y.append(alpha1 - 0.5 * width - x[i])
		if x[i] <= alpha1 and x[i] >=alpha1 - width:
			y.append(0.5/width*(x[i] - alpha1)*(x[i] - alpha1))
	return np.array(y)

# 100 linearly spaced numbers
x = np.linspace(-2,7,100)
alpha1 = 3
alpha2 = 4
width = 1.5

y = hinge1(x,alpha1)
z= smoothhinge1(x, alpha1, width)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.spines['left'].set_position('center')
#ax.spines['bottom'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

# plot the functions
plt.plot(x,y, 'b', label='max(0, alpha1-x)')
plt.plot(x,z, 'r', label='V1')
plt.xlabel("x")
plt.ylabel("y")

plt.legend(loc='upper left')

# show the plot
plt.show()