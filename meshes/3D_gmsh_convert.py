"""
Step 1. convert the mesh to.xdmf  format using meshio
"""

#Pass filename without .msh as argument
import sys
 
filename = str(sys.argv[1])
filenamemsh = (filename + ".msh")
filenamexdmf = (filename + ".xdmf")
filenamexdmf_facet = ("facet_" + filename + ".xdmf")

#filename  = input("Please enter file name (without .msh): ")
#filenamemsh = (filename + ".msh")
import meshio
#mesh_from_file = meshio.read("Sphere_Quadrant.msh")
mesh_from_file = meshio.read(filenamemsh)

"""
Step. 2  Extract cells and boundary data.

Now that we have created the mesh, we need to extract the cells 
and physical data. We need to create a separate file for the 
facets (lines),  which we will use when we define boundary 
conditions in  Fenics. We do this  with the following convenience 
function. Note that as we would like a  2 dimensional mesh, we need to 
remove the z-values in the mesh coordinates, if any.
"""
import numpy
def create_mesh(mesh, cell_type, prune_z=False):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    points = mesh.points[:,:2] if prune_z else mesh.points
    out_mesh = meshio.Mesh(points=points, cells={cell_type: cells},\
                           cell_data={"name_to_read":[cell_data]})
    return out_mesh

"""
Step 3.
With this function in hand, we can save the facet line mesh 
and the domain triangle  mesh in `XDMF` format 
"""

triangle_mesh = create_mesh(mesh_from_file, "triangle", prune_z=False)
meshio.write(filenamexdmf_facet, triangle_mesh)

tetra_mesh = create_mesh(mesh_from_file, "tetra", prune_z=False)
meshio.write(filenamexdmf, tetra_mesh)

