# -*- coding:utf-8 -*-
# Copyright© by Yehui Zhang
import numpy as np
from numpy import linspace, eye, pi, array, sin, cos
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from copy import deepcopy

## Pre configure
PreSwitch = False
EasyAxis = "Z"
# DPI for save figure
DPI = 100
# Magnetic atom number
Atom = 2
# matplotlib: https://matplotlib.org/examples/color/colormaps_reference.html
# prefix=plt.cm
colormap = plt.cm.Reds_r
# 'Elev' stores the elevation angle in the z plane.
# 'Azim' stores the azimuth angle in the x,y plane.
Elev, Azim = 20, 40
# True:R; False:1(constant)
R_flag = True
# Fitting surface by numpy
Fit_flag = True
# Transparency
Alpha = 0.8
# Font for title
fonts = "Consolas"


def plot_init(Energy):
    # Plot the surface.
    fig = plt.figure(figsize=(8, 6))
    ax = fig.gca(projection='3d')
    ax.view_init(elev=Elev, azim=Azim)
    for limit in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        limit(-0.8, 0.8)

    # Close plane & background
    ax._axis3don = False
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)

    # Add a color bar which maps values to colors.
    Mappable = plt.cm.ScalarMappable(cmap=colormap)
    Mappable.set_array(Energy)
    ticks = linspace(0, np.max(Energy), 8, dtype='int64')
    cb = plt.colorbar(Mappable, shrink=0.5, aspect=25, ticks=ticks, extend='both')
    cb.ax.set_title(unicode("$MAE_{[µeV/atom]}$", 'utf-8'), family=fonts, size=12)

    return fig, ax


def Draw_axis(ax, alpha):
    tx, ty, tz = eye(3) * -1.4
    lx, ly, lz = eye(3) * 2.8
    colorset = (0, 0, 0, alpha)
    ax.quiver(tx, ty, tz, lx, ly, lz, arrow_length_ratio=0.05, color=colorset)
    for ((cx, cy, cz), axis) in zip(eye(3) * 1.5, ['x', 'y', 'z']):
        ax.text(cx, cy, cz, axis, family=fonts, size=25, ha='center', va='center', color=colorset)
    return


# Read file
Data = open("Data.dat", "r").readlines()
flag = int(Data[0].split()[2])

# Make mesh
Num_longitude, Num_latitude = 397, 199
longitude = linspace(0, 2 * pi, Num_longitude)
latitude = linspace(0, pi, Num_latitude)
line = linspace(0, pi / 2, flag)
SAXIS = [[0, i, 1, sin(i), 0, cos(i)] for i in line]
SAXIS += [[pi / 2, i, 1, 0, sin(i), cos(i)] for i in line]
SAXIS += [[i, 0, 1, sin(i), cos(i), 0] for i in line]
[SAXIS[i].append(float(Data[i + 2].split()[10])) for i in xrange(len(SAXIS))]
SAXIS = array(SAXIS)

if PreSwitch:
    R = SAXIS[:, 6] / np.max(SAXIS[:, 6])
    X, Y, Z = SAXIS[:, 3] * R, SAXIS[:, 4] * R, SAXIS[:, 5] * R
    colors = colormap(R, Alpha)
    fig, ax = plot_init(SAXIS[:, 6])
    perm = [[i, j, k] for i in [1, -1] for j in [1, -1] for k in [1, -1]]
    for (i, j, k) in perm:
        ax.scatter(i * X, j * Y, k * Z)
    plt.show()
    exit()

# Grep Energy and sort and fit energy data
Energy = longitude.reshape(-1, 1) + latitude
if EasyAxis == "X":
    tmp = deepcopy([i for i in SAXIS[:flag]])
    for i in SAXIS[:flag][-2::-1]:
        tmp.append(i)
        tmp[-1][3] *= -1
    tmp = array(tmp)
    para = np.polyfit(tmp[:, 3], tmp[:, 6], 7)
    for i in xrange(Num_longitude):
        for j in xrange(Num_latitude):
            x = abs(sin(latitude[j]) * cos(longitude[i]))
            Energy[i][j] = np.polyval(para, x) / Atom

if EasyAxis == "Y":
    tmp = deepcopy([i for i in SAXIS[flag:2 * flag]])
    for i in SAXIS[flag:2 * flag][-2::-1]:
        tmp.append(i)
        tmp[-1][4] *= -1
    tmp = array(tmp)
    para = np.polyfit(tmp[:, 4], tmp[:, 6], 7)
    for i in xrange(Num_longitude):
        for j in xrange(Num_latitude):
            y = abs(sin(latitude[j]) * sin(longitude[i]))
            Energy[i][j] = np.polyval(para, y) / Atom

if EasyAxis == "Z":
    tmp = deepcopy([i for i in SAXIS[:flag]])
    for i in SAXIS[:flag][-2::-1]:
        tmp.append(i)
        tmp[-1][5] *= -1
    tmp = array(tmp)
    para = np.polyfit(tmp[:, 5], tmp[:, 6], 7)
    for i in xrange(Num_longitude):
        for j in xrange(Num_latitude):
            z = abs(cos(latitude[j]))
            Energy[i][j] = np.polyval(para, z) / Atom

# Set R value
NormEnergy = Energy / np.max(Energy)
R = (NormEnergy if R_flag else 1)

# Make data.
x = cos(longitude.reshape(-1, 1)) * sin(latitude) * R
y = sin(longitude.reshape(-1, 1)) * sin(latitude) * R
z = 0 * longitude.reshape(-1, 1) + cos(latitude) * R
colors = colormap(NormEnergy, Alpha)

## Plot without axis
fig, ax = plot_init(Energy)
# Add plot surface
ax.plot_surface(x, y, z, facecolors=colors, linewidth=0, antialiased=True)
# Draw axis arrows
Draw_axis(ax, 0)
# Save figure
plt.savefig("output_part.png", format='png', bbox_inches='tight', dpi=DPI)

## Plot equator
plt.close()
fig, ax = plot_init(Energy)
# Draw axis arrows
Draw_axis(ax, 1)
ax.plot(x[:, (Num_latitude - 1) / 2], y[:, (Num_latitude - 1) / 2], z[:, (Num_latitude - 1) / 2])
# Save figure
plt.savefig("output_zero.png", format='png', bbox_inches='tight', dpi=DPI)

## Plot ALL
plt.close()
fig, ax = plot_init(Energy)
# Add plot surface
ax.plot_surface(x, y, z, color=colormap(1, Alpha), facecolors=colors, linewidth=0, antialiased=True)
# Draw axis arrows
Draw_axis(ax, 1)
# Save figure
plt.savefig("output.png", format='png', bbox_inches='tight', dpi=DPI)
plt.show()