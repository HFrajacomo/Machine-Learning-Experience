#
import numpy
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import ListedColormap, BoundaryNorm
from sklearn import neighbors
import matplotlib.patches as mpatches
import graphviz
from sklearn.tree import export_graphviz
import matplotlib.patches as mpatches
import sys

def setList(lista):
    tmp = []
    for element in lista:
        if(element not in tmp):
            tmp.append(element)
    return tmp

def orderList(lista):
    tmp = []
    end = 0
    i = 0
    for element in lista:
        end = len(tmp)
        if(len(tmp) == 0):
            tmp.append(element)
        elif(tmp[end-1] <= element):
            tmp.append(element)
        else:
            for i in range (0,len(tmp)):
                if(tmp[i] == element):
                    tmpo = tmp[:i] + [element] + tmp[i:]
                    tmp = tmpo
                    break
                elif(tmp[i]>element):
                    tmpo = tmp[:i-1] + [element] + tmp[i-1:]
                    tmp = tmpo
                    break
    return tmp


def miraplot_knn(X, y, n_neighbors, weights, names, X_label, Y_label):
    #X_mat = X[['height', 'width']].as_matrix()

    X_mat = X[[list(X)[X_label], list(X)[Y_label]]].as_matrix()

    y_mat = y.as_matrix()
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF','#AFAFAF','#0FF5AA', '#552255', '#837065'])
    cmap_bold  = ListedColormap(['#FF0000', '#00FF00', '#0000FF','#AFAFAF','#0FF5EE', '#550055', '#203021'])

    clf = neighbors.KNeighborsClassifier(n_neighbors, weights=weights)
    clf.fit(X_mat, y_mat)


    # Plot the decision boundary by assigning a color in the color map
    # to each mesh point.
    
    mesh_step_size = 1  # step size in the mesh  0.01
    plot_symbol_size = 50
    

    x_min, x_max = X_mat[:, 0].min() - 1, X_mat[:, 0].max() + 1
    y_min, y_max = X_mat[:, 1].min() - 1, X_mat[:, 1].max() + 1
    xx, yy = numpy.meshgrid(numpy.arange(x_min, x_max, mesh_step_size),
                         numpy.arange(y_min, y_max, mesh_step_size))
    Z = clf.predict(numpy.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot training points
    plt.scatter(X_mat[:, 0], X_mat[:, 1], s=plot_symbol_size, c=y, cmap=cmap_bold, edgecolor = 'black')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    # Plot Loop

    Dictionary = {"pistol": 0, "shotgun": 1, "submachine": 2, "rifle": 3, "heavy": 4, "sniper": 5, "null": 6}        

    if(len(names) == 2):
        patch0 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[0]]], label=str(names[0]))
        patch1 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[1]]], label=str(names[1]))
        plt.legend(handles=[patch0, patch1])
    if(len(names) == 3):    
        patch0 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[0]]], label=str(names[0]))
        patch1 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[1]]], label=str(names[1]))
        patch2 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[2]]], label=str(names[2]))         
        plt.legend(handles=[patch0, patch1, patch2])     
    if(len(names) == 4):
        patch0 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[0]]], label=str(names[0]))
        patch1 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[1]]], label=str(names[1]))
        patch2 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[2]]], label=str(names[2]))
        patch3 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[3]]], label=str(names[3]))           
        plt.legend(handles=[patch0, patch1, patch2])
    if(len(names) == 5):
        patch0 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[0]]], label=str(names[0]))
        patch1 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[1]]], label=str(names[1]))
        patch2 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[2]]], label=str(names[2]))
        patch3 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[3]]], label=str(names[3]))           
        patch4 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[4]]], label=str(names[4]))       
        plt.legend(handles=[patch0, patch1, patch2, patch3, patch4])
    if(len(names) == 6):
        patch0 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[0]]], label=str(names[0]))
        patch1 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[1]]], label=str(names[1]))
        patch2 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[2]]], label=str(names[2]))
        patch3 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[3]]], label=str(names[3]))           
        patch4 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[4]]], label=str(names[4]))       
        patch5 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[5]]], label=str(names[5]))      
        plt.legend(handles=[patch0, patch1, patch2, patch3, patch4, patch5])  
    if(len(names) == 7):
        patch0 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[0]]], label=str(names[0]))
        patch1 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[1]]], label=str(names[1]))
        patch2 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[2]]], label=str(names[2]))
        patch3 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[3]]], label=str(names[3]))           
        patch4 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[4]]], label=str(names[4]))       
        patch5 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[5]]], label=str(names[5]))      
        patch6 = mpatches.Patch(color=cmap_bold.colors[Dictionary[names[6]]], label=str(names[6]))
        plt.legend(handles=[patch0, patch1, patch2, patch3, patch4, patch5, patch6])
    if(len(names)<=1):
        sys.exit("The max amount of classes has to be over 1")
    if(len(names)>7):
        sys.exit("The max amount of classes has to be under 7")
        
    plt.xlabel(str(list(X)[X_label]))
    plt.ylabel(str(list(X)[Y_label]))

    plt.show()
