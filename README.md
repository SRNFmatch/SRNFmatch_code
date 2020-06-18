SRNFmatch
=========

Description
-----------

This Python package provides a set of tools for the comparison, matching and interpolation of triangulated surfaces within the elastic shape analysis setting. It allows specifically to solve the geodesic matching and distance computation problem between two surfaces with respect to the square root normal field (SRNF) or square root curvature field (SRCF) metric. Such distances can in turn be used for a variety of tasks such as shape classification, clustering, regression... Unlike previous works on the topic, our variational approach involves relaxing the matching constraint based on a varifold discrepancy term which makes the method flexible to deal with discrete surfaces of variable sampling and/or topology. SRNFmatch also provides functions to approximate the inverse of an SRNF map or to estimate the geodesic path between a surface and its deformation, as well as various utilities to export and visualize results.    

More details can be found in the reference article given below.



References
------------


    @misc{BCHH2020,
      author  = {Martin Bauer, Nicolas Charon, Philipp Harms and Hsi-Wei Hsieh},
      title   = {A numerical framework for elastic surface matching, comparison, and interpolation},
      note    = {Preprint available on ArXiv},
      year    = {2020},
    }

Please cite this paper in your work.



Dependencies
------------

SRNFmatch is entirely written in Python while taking advantage of parallel computing on GPUs through CUDA. 
For that reason, it must be used on a machine equipped with an NVIDIA graphics card with recent CUDA drivers installed.
The code involves in addition the following Python libraries:

* Numpy and Scipy
* Pytorch 
* Keops (https://www.kernel-operations.io/keops/index.html)
* PyVTK (https://pypi.org/project/PyVTK/)

Note that PyVTK is primarily used for surface reading, saving, visualization and simple mesh processing operations (decimation, subdivision...). Other libraries such as PyMesh could be used as potential replacement with relatively small modifications to our code.  


Usage
-----

See the two script files in the "Demo scripts" folder for some examples of basic use of the code. 
The first script "script_demo_multires.py" computes the SRNF matching between two cup surfaces (c.f Fig. 7 of the above reference) with the multiresolution version of the algorithm.
The second script "script_demo_texture.py" illustrates the ability to incorporate surface texture as additional information and can be used to reproduce Fig. 9 of the above reference. 


Licence
-------

This program is free software: you can redistribute it and/or modify it under 
the terms of the GNU General Public License as published by the Free Software 
Foundation, either version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT 
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS 
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program. If not, see http://www.gnu.org/licenses/.


Contacts
--------

* Martin Bauer (bauer at math dot fsu dot edu)
* Nicolas Charon (charon at cis dot jhu dot edu)
* Philipp Harms (philipp dot harms at stochastik dot uni-freiburg dot de)
