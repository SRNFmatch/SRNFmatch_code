import sys
sys.path.append('../src/')
import matching

import input_output

# Load Data
[VS,FS,FunS] = input_output.loadData("../Data/cup2.ply")
source = [VS,FS]

[VT,FT,FunT] = input_output.loadData("../Data/cup3_broken.ply")
target= [VT,FT]

# Decimate source mesh to compute initialization for the multires algorithm 
param_decimation = {'factor':15/16,'Vol_preser':1, 'Fun_Error_Metric': 1, 'Fun_weigth':0.00} #decimate by a factor of 16
[V0,F0]= input_output.decimate_mesh(VS,FS,param_decimation)
source_init = [V0,F0]
target_init = [V0,F0]

# Set parameters for the two successive runs
weight_MCV=0 #no SRCF term

parameters1 = {'weight_MCV':weight_MCV,'weight_coef_dist_S': 10**2, 'weight_coef_dist_T': 10**2,\
          'kernel_geom':'gaussian','sig_geom':.15,'kernel_grass':'binet',\
          'sig_grass':1 ,'max_iter': 300,'use_fundata':0}
parameters2 = {'weight_MCV':weight_MCV,'weight_coef_dist_S': 10**2, 'weight_coef_dist_T': 10**2,\
          'kernel_geom':'gaussian','sig_geom': .1,'kernel_grass':'binet',\
          'sig_grass': 1,'max_iter': 300,'use_fundata':0}
paramlist=[parameters1,parameters2]


# Run multiresolution matching 
f0,f1,Tri,Funct,En,Dic = matching.MultiResMatching(source,target,source_init,target_init,paramlist)

match_source=[f0[1],Tri[1]]
match_target=[f1[1],Tri[1]]

# Compute estimated squared SRV distance
dist = matching.SRVF_dist_square([f0[1],Tri[1]],[f1[1],Tri[1]])

# Plot matching result (press q to exit window)
iren, renWin = input_output.plotMatchingResult(source,match_target,target,'Symmetric',match_source)
input_output.close_vtk_interactor(iren)
del iren, renWin

# Compute geodesic 
Geod = matching.SRNF_geodesic(match_source,match_target,4,500)

# Plot geodesic 
iren, renWin = input_output.plotGeodesic(Geod,Tri[2])
input_output.close_vtk_interactor(iren)
del iren, renWin