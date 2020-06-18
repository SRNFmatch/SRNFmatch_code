import sys
sys.path.append('../src/')
import matching

import input_output

# Load Data
VS,FS,FunS = input_output.loadData("../Data/sphere_colored.mat")
source = [VS,FS,FunS]

VT,FT,FunT = input_output.loadData("../Data/sphere_single_bump_colored.mat")
target= [VT,FT,FunT]

# Plot source surface (press q to exit window)
iren, renWin = input_output.plotMesh(source) 
input_output.close_vtk_interactor(iren)
del iren, renWin

# Plot target surface (press q to exit window)
iren, renWin = input_output.plotMesh(target) 
input_output.close_vtk_interactor(iren)
del iren, renWin

# Choose an initialization for the algorithm
target_init = [VS,FS,FunS]

# Set parameters
parameters = {'weight_MCV':0,'weight_coef_dist_T': 10**3,\
          'kernel_geom':'gaussian','sig_geom':.2,'kernel_grass':'binet',\
          'sig_grass':1 ,'kernel_fun':'gaussian','sig_fun':0.3,'max_iter': 500,'use_fundata':1}

# Run standard matching 
f1,En,Dic = matching.StandardMatching(source,target,target_init,parameters)

# Compute estimated squared SRV distance
dist = matching.SRVF_dist_square([VS,FS],[f1,FS])

# Plot matching result (press q to exit window)
match=[f1,FS,FunS]
iren, renWin = input_output.plotMatchingResult(source,match,target)
input_output.close_vtk_interactor(iren)
del iren, renWin

# Compute geodesic 
match = [f1,FS]
Geod = matching.SRNF_geodesic(source,match,5,10000)

# Plot geodesic 
iren, renWin = input_output.plotGeodesic(Geod,FS,FunS)
input_output.close_vtk_interactor(iren)
del iren, renWin
