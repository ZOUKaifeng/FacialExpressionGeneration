#!/usr/bin/env python3

from matplotlib.widgets import Slider  # import the Slider widget
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import sys

import tkinter as tk
from tkinter.filedialog import askopenfilename

filename = askopenfilename()


# Load the data
global allXyz
allXyz=np.load(filename)
numRow,numCol=allXyz.shape

global shauteur

def press(event):
    global shauteur
    print('press', event.key)
    sys.stdout.flush()
    if event.key == 'right':
        if shauteur.val < shauteur.valmax: 
            shauteur.set_val(shauteur.val+1)
    if event.key == 'left':
        if shauteur.val > shauteur.valmin: 
            shauteur.set_val(shauteur.val-1)
        

# Create the window
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
fig.canvas.mpl_connect('key_press_event', press)

        
# Show one frame (one line of the xyz matrix)
def showLine(rowID):
    global allXyz
    x =list(allXyz[rowID][0::3])
    y =list(allXyz[rowID][1::3])
    z =list(allXyz[rowID][2::3])
    pg = ax.scatter(x, y, z, c='r', marker='o')
    print("Frame: " + str(rowID))
    return pg


global pointGraph
pointGraph = showLine(1)

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

axhauteur = plt.axes([0.2, 0.1, 0.65, 0.03])
shauteur = Slider(axhauteur, 'Hauteur', 0, numRow-1, valinit=0, valfmt='%0.0f')


def update(val): 
    global pointGraph
    # Remove the previous point
    pointGraph.remove()
    # Add the new points
    pointGraph = showLine(int(val))
    fig.canvas.draw_idle()
shauteur.on_changed(update)

# Show the window
plt.show()

