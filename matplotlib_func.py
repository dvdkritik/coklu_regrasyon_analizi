import matplotlib.pyplot as plt 
import numpy as np

# Data for plotting
t = np.arange(0.0, 2.0, 0.01) #creates set of 200 value from 0.01,0.02...1.99
s = 1 + np.sin(2 * np.pi * t) #Sine wave 2*pi*t

# Note that using plt.subplots below is equivalent to using
# fig = plt.figure() and then ax = fig.add_subplot(111)
fig, ax = plt.subplots()
ax.plot(t, s)

#label and title 
ax.set(xlabel='time (s)', ylabel='voltage (mV)',
       title='About as simple as it gets, folks')
ax.grid()


#save as image
fig.savefig("test.png") 

#show the plot
plt.show()
