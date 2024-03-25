This repository contains codes and supplementary videos relating to Chockalingam Senthilnathan's PhD thesis. The following resources were used during the development of these codes:

1. Anand, L.: Introduction to Coupled Theories in Solid Mechanics. Unpublished MIT 2.077 course notes (2023).
2. E. M. Stewart and L. Anand. Example codes for coupled theories in solid mechanics, 2023. URL https://github.com/SolidMechanicsCoupledTheories/example_codes.

SwellingGrowth.py has the FeniCS implementation of the swelling growth theory. It is clearly commented and all the run options are parameterized.
Inclusion_Growth.py has the FeniCS implementation of the inclusion morphogenesis problem.
Void_Growth.py has the FeniCS implementation for pressurized expansion of a void.
All the mesh files required are available in the meshes folder. The meshes have already been generated. 
But if you wish to generate them, these are the steps
1. First generate .msh file from the .geo file using gmsh For example - 
a) gmsh Ellip_Void_EllipBoundary_BbyA100_Phi05_res_pt04.geo -2
or b) gmsh Box_Wall3D.geo -3
2. Once you generat the msh file, use the python scripts to convert to xdmf files. Use the appropriate script depending on 2D or 3D geometry and whether or not the geometry has facets. For example
a) python 2d_gmsh_convert.py Ellip_Void_EllipBoundary_BbyA100_Phi05_res_pt04
b) python 3D_gmsh_convert_nofacets.py  Box_wall3D

All results in my thesis should be generatable from the files here

Supplementary videos are available in the Supplementary Videos folder.
