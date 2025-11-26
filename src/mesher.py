import numpy as np
import matplotlib.pyplot as plt

# Generate mesh grid in 3D space (x, y, z)

Nx = 256*2; Ny = 128; Nz = 128

# y and z are uniform grids
y = np.linspace(0,2.0*np.pi,Ny)
z = np.linspace(0,2.0*np.pi,Nz)

# x is a non-uniform grid, 
