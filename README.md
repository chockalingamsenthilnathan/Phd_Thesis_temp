This repository contains codes and supplementary videos relating to Chockalingam Senthilnathan's PhD thesis. The following resources were used during the development of these codes:

1. Anand, L.: Introduction to Coupled Theories in Solid Mechanics. Unpublished MIT 2.077 course notes (2023).
2. E. M. Stewart and L. Anand. Example codes for coupled theories in solid mechanics, 2023. URL https://github.com/SolidMechanicsCoupledTheories/example_codes.

SwellingGrowth.py has the FEniCS implementation of the swelling growth theory. It is clearly commented and all the run options are parameterized.
Inclusion_Growth.py has the FEniCS implementation of the inclusion morphogenesis problem.
Void_Growth.py has the FEniCS implementation for pressurized expansion of a void.
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

If you use these codes in your own research, cite the following:

1. Chockalingam, S. and Cohen, T. FEniCS codes for a large deformation swelling-growth theory, 2024. URL https://github.com/chockalingamsenthilnathan/Phd_Thesis_temp 
2. Chockalingam, S., & Cohen, T. (2024). [A large deformation theory for coupled swelling and growth with application to growing tumors and bacterial biofilms.](https://www.sciencedirect.com/science/article/pii/S0022509624000930) Journal of the Mechanics and Physics of Solids, 105627.
3. Chockalingam, S. and Cohen, T., 2024, Explaining morphogenesis under mechanical confinement using a large deformation swelling-growth theory (in preparation.) 
4. Anand, L.: Introduction to Coupled Theories in Solid Mechanics. Unpublished MIT 2.077 course notes (2023).
5. E. M. Stewart and L. Anand. Example codes for coupled theories in solid mechanics, 2023. URL https://github.com/SolidMechanicsCoupledTheories/example_codes.
