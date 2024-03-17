"""
Problem: Growing elliptical void (pressure controlled and volume calculated) against
a stiff medium. Axisymmetric and 2D plane strain implementations included.

Dimensionless version of theory is implemented in the code

Degrees of freedom:
#
Displacement: ubar
pressure: pbar

Contact:
Chockalingam
chocka94@mit.edu
chocka94@gmail.com

Mar 2024
"""

from ufl import * #Needed for tanh() call if required
# Fenics-related packages
from dolfin import *
# Numerical array package
import numpy as np
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime

import pickle
import scipy.io

# Set level of detail for log messages (integer)
# 
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
#
set_log_level(30)

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4     

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

b0 = 1.0 # Ellipse undeformed minor axis
Phi0 = 5.0 # Ellipse undeformed apect ratio a/b 
a0 = Phi0*b0 # Ellipse undeformed minor axis

BbyA = 100.0 #Outer Ellipse dimension ratio

#This file is not set up in parts for full 3D simulations, choose onl1 option 2 or 3
simulation_type = 2 #2 is axiysymmetric/ellipsoid, 3 is plane strain/planar ellipse

# Initialize an empty mesh object
mesh = Mesh()


meshname = "meshes/Ellip_Void_EllipBoundary_BbyA{0:d}_Phi0{1:d}".format(int(BbyA), int(Phi0)) + "_res_pt04.xdmf"
facet_meshname = "meshes/facet_Ellip_Void_EllipBoundary_BbyA{0:d}_Phi0{1:d}".format(int(BbyA), int(Phi0)) + "_res_pt04.xdmf"

# Read the .xdmf  file data into mesh object
with XDMFFile(meshname) as infile:
    infile.read(mesh)
    
# Extract initial mesh coords
x = SpatialCoordinate(mesh)
    
# Read mesh boundary facets.
# Specify parameters of 1-D mesh function object
mvc_1d = MeshValueCollection("size_t", mesh, 1)
# Read the   *.xdmf  data into mesh function object
with XDMFFile(facet_meshname) as infile:
    infile.read(mvc_1d, "name_to_read")
    
# Store boundary facets data
facets = cpp.mesh.MeshFunctionSizet(mesh, mvc_1d)

# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)
n = FacetNormal(mesh) 
#dx = Measure('dx', domain=mesh, subdomain_data=mf)

#From GMSh
#Physical Curve("bottom", 14) 
#Physical Curve("outer", 15) 
#Physical Curve("left", 17) 
#Physical Curve("inclusion_curve", 18) 
surface_boundary_number = 18

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
KbyG = Constant(1000)

"""
Simulation time-control related params
"""
tdummy = 0.0
steps = 50 #1000 #Run at high resolution for accuracy
dt = 1.0/steps #Initial time stepping size, adaptive solver can change

vis_steps = 5 #print detailed results for every vis_steps steps

adaptive_solver = 1 #Adaptive time stepping if value is 1
adaptive_solver_iters_min = 2
adaptive_solver_iters_max = 15
adaptive_solver_cutbacklimit = 3
adaptive_solver_decreasefactor = 2
adaptive_solver_increasefactor = 1.5

#max applied pressure Pbar
pbarb_max = 2.2 #In axisymmetric case, 2.5 is the cavitation limit

pbar_b = Expression(("pbarb_max*tdummy"),
                     pbarb_max = pbarb_max, tdummy = tdummy, degree=1)    
 
'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For ubar
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For  pbar

#
TH = MixedElement([U2, P1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)    # Total space for all DOFs

# Define trial functions
w = Function(ME)
ubar, pbar = split(w)

# A copy of functions to store values in the previous step
w_old = Function(ME)
ubar_old,   pbar_old = split(w_old)   

# Define test functions
w_test = TestFunction(ME)                
ubar_test, pbar_test = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)             

''''''''''''''''''''''
SUBROUTINES
'''''''''''''''''''''
'''
For plane strain:
'''

#DEFORMATION GRADIENT

# Axisymmetric simulations
def F_axi_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F = Id + grad(u)            # 2D Deformation gradient
    F33_exp =  (x[0]+u[0])/x[0]  # axisymmetric F33, R/R0 
    F33 = conditional(lt(x[0], DOLFIN_EPS), 1.0, F33_exp) # avoid divide by zero at r=0 
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, F33]]) # Full axisymmetric F

#Full 3D simulations
def F_calc3D(u):
    Id = Identity(3) 
    F  = Id + grad(u)
    return F

# Plane strain deformation gradient 
def F_pe_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F  = Id + grad(u)            # 2D Deformation gradient
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, 1]]) # Full pe F

def F_calc(u):
    if(simulation_type==1): #Full 3D
        return F_calc3D(u)
    elif(simulation_type==2):
        return F_axi_calc(u)
    elif(simulation_type==3):
        return F_pe_calc(u)

#2D submatrix of F 

def F_axi_calc2D(u): #Axisymmetric
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F  = Id + grad(u)            # 2D Deformation gradient
    return as_tensor([[F[0,0], F[0,1]],
                  [F[1,0], F[1,1]]])

def F_pe_calc2D(u): #Plane strain
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F  = Id + grad(u)            # 2D Deformation gradient
    return as_tensor([[F[0,0], F[0,1]],
                  [F[1,0], F[1,1]]]) 

def F_calc2D(u):
    if(simulation_type==2): 
        return F_axi_calc2D(u)
    elif(simulation_type==3):
        return F_pe_calc2D(u)

#VECTOR FIELD GRADIENT

def axi_grad_vector(u): #Axisymmetric
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, u[0]/x[0]]]) 

def pe_grad_vector(u): #Plane Strain
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, 0]]) 

def grad_vector(u):
    if(simulation_type==1): #Full 3D
        return grad(u)
    elif(simulation_type==2):
        return axi_grad_vector(u)
    elif(simulation_type==3):
        return pe_grad_vector(u)

# SCALAR FIELD GRADIENT
    
def axi_grad_scalar(y): #Axisymmetric
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

def pe_grad_scalar(y): #Plane strain
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

def grad_scalar(y):
    if(simulation_type==1): #Full 3D
        return grad(y)
    elif(simulation_type==2):
        return axi_grad_scalar(y)
    elif(simulation_type==3):
        return pe_grad_scalar(y)    

#  Elastic Je
def Je_calc(u,Js,Jg):
    F = F_calc(u)  
    detF = det(F)   
    #
    Je    = (detF/Js/Jg)     #Js and Jg are 1 in this script
    return   Je    

# Normalized Piola stress for Arruda_Boyce material
def Piola_calc(u,pbar):
    F     = F_calc(u)
    J = det(F)
    Tmatbar = (F - inv(F.T) ) - J*pbar*inv(F.T)
    return Tmatbar

#-----------------------------------------------------------------------------

'''''''''''''''''''''''''''''
Kinematics and constitutive relations
'''''''''''''''''''''''''''''
# Kinematics
F = F_calc(ubar)
J = det(F)  # Total volumetric jacobian
C = (F.T)*F

Fcof = J*inv(F.T)


if(simulation_type!=1):
    # Cofactor of F
    F2D = F_calc2D(ubar);
    C2D = (F2D.T)*(F2D)
    J2D = det(F2D)
    F2Dcof = J2D*inv(F2D.T)
    Fcof_2D = as_tensor([[Fcof[0,0], Fcof[0,1]],
              [Fcof[1,0], Fcof[1,1]]]) #DIFFERENT FROM F2dcof for axisym


# Elastic volumetric Jacobian
Je = Je_calc(ubar,1.0,1.0)              #Je is trivially J in this case       

#  Normalized Piola stress
Tmatbar = Piola_calc(ubar, pbar)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: ubar)
# Res_1: Pressure variable (test fxn: pbar)

# Time step field
dt_dummy = Constant(dt)

# Configuration-dependent traction
if(simulation_type==1):
    traction = -pbar_b*dot(Fcof,n)
else:
    traction = -pbar_b*dot(Fcof_2D,n)

# The weak form for the balance of forces
if(simulation_type==2):
    Res_0 =  inner(Tmatbar, grad_vector(ubar_test) )*x[0]*dx  \
                - dot(traction, ubar_test)*x[0]*ds(surface_boundary_number) 
else:
    Res_0 =  inner(Tmatbar, grad_vector(ubar_test) )*dx  \
                - dot(traction, ubar_test)*ds(surface_boundary_number)                  

# The weak form for the auxiliary pressure variable definition
if(simulation_type==2):
    Res_1 = dot((pbar*Je/KbyG + ln(Je)) , pbar_test)*x[0]*dx   
else:
    Res_1 = dot((pbar*Je/KbyG + ln(Je)) , pbar_test)*dx  
         
# Total weak form
Res = Res_0 + Res_1 

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

'''''''''''''''''''''''
Dirichlet  Boundary Conditions
'''''''''''''''''''''''      
# Gmsh labels
#Physical Curve("bottom", 14) 
#Physical Curve("right", 15)
#Physical Curve("top", 16)
#Physical Curve("left", 17) 
#Physical Curve("inclusion_curve", 18) 

bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 17)  # u1 fix - left  
bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 14)  # u2 fix - bottom
# 
bcs = [bcs_1, bcs_2]

"""
SETUP NONLINEAR PROBLEM
"""
GelProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)
solver  = NonlinearVariationalSolver(GelProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps" 
prm['newton_solver']['absolute_tolerance'] = 1.e-8
prm['newton_solver']['relative_tolerance'] = 1.e-8
prm['newton_solver']['maximum_iterations'] = 30

'''''''''''''''''''''
 SET UP OUTPUT FILES
'''''''''''''''''''''

if(simulation_type==2):
    savename = ("results/Axisym_UnifVGrow_Void/EllipB_Phi0{0:d}_BbyA{1:d}_respt04_pb{2:.2f}_Steps{3:.1f}" \
            .format(int(Phi0),int(BbyA),pbarb_max,steps))
elif(simulation_type==3):
    savename = ("results/2DPlaneStrain_UnifVGrow_Void/EllipB_Phi0{0:d}_BbyA{1:d}_respt04_pb{2:.2f}_Steps{3:.1f}" \
            .format(int(Phi0),int(BbyA),pbarb_max,steps))        

savenamexdmf = savename + ".xdmf"    
# Output file setup
file_results = XDMFFile(savenamexdmf)
# "Flush_output" permits reading the output during simulation
# (Although this causes a minor performance hit)
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Function space for projection of results
W2 = FunctionSpace(mesh, U2) # Vector space for visualization  
W = FunctionSpace(mesh,P1)   # Scalar space for visualization 
     
def writeResults(t):
      # Variable projecting and renaming
      ubar_Vis = project(ubar, W2)
      ubar_Vis.rename("dispbar"," ")
      
      # Visualize the pressure
      pbar_Vis = project(pbar, W)
      pbar_Vis.rename("pbar"," ")
      
      # Visualize J
      J_Vis = project(J, W)
      J_Vis.rename("J"," ")    
      
      # Visualize Je
      Je_Vis = project(Je, W)
      Je_Vis.rename("Je"," ")    
      
      # Visualize some components of Piola stress
      P11bar_Vis = project(Tmatbar[0,0],W)
      P11bar_Vis.rename("P11bar","")
      P22bar_Vis = project(Tmatbar[1,1],W)
      P22bar_Vis.rename("P22bar","")    
      P33bar_Vis = project(Tmatbar[2,2],W)
      P33bar_Vis.rename("P33bar","")  
      
      P12bar_Vis = project(Tmatbar[0,1],W)
      P12bar_Vis.rename("P12bar","")
      P13bar_Vis = project(Tmatbar[0,2],W)
      P13bar_Vis.rename("P13bar","")    
      P23bar_Vis = project(Tmatbar[1,2],W)
      P23bar_Vis.rename("P23bar","")        
      
      P21bar_Vis = project(Tmatbar[1,0],W)
      P21bar_Vis.rename("P21bar","")
      P31bar_Vis = project(Tmatbar[2,0],W)
      P31bar_Vis.rename("P31bar","")    
      P32bar_Vis = project(Tmatbar[2,1],W)
      P32bar_Vis.rename("P23bar","")   
      
      # Visualize the Mises stress  & Cauchy stress
      Tbar    = Tmatbar*F.T/J
      T11bar_Vis = project(Tbar[0,0],W)
      T11bar_Vis.rename("T11bar","")
      T22bar_Vis = project(Tbar[1,1],W)
      T22bar_Vis.rename("T22bar","")    
      T33bar_Vis = project(Tbar[2,2],W)
      T33bar_Vis.rename("T33bar","")  
      
      T12bar_Vis = project(Tbar[0,1],W)
      T12bar_Vis.rename("T12bar","")
      T13bar_Vis = project(Tbar[0,2],W)
      T13bar_Vis.rename("T13bar","")    
      T23bar_Vis = project(Tbar[1,2],W)
      T23bar_Vis.rename("T23bar","")        
      
      T21bar_Vis = project(Tbar[1,0],W)
      T21bar_Vis.rename("T21bar","")
      T31bar_Vis = project(Tbar[2,0],W)
      T31bar_Vis.rename("T31bar","")    
      T32bar_Vis = project(Tbar[2,1],W)
      T32bar_Vis.rename("T23bar","")   
      
      
      # Visualize the non bulk modulus stress
      Tshbar = Tbar + pbar*Identity(3)
      Tsh11bar_Vis = project(Tshbar[0,0],W)
      Tsh11bar_Vis.rename("Tsh11bar","")
      Tsh22bar_Vis = project(Tshbar[1,1],W)
      Tsh22bar_Vis.rename("Tsh22bar","")    
      Tsh33bar_Vis = project(Tshbar[2,2],W)
      Tsh33bar_Vis.rename("Tsh33bar","")   
      
      tr_Tbar_Vis = project(tr(Tbar),W)
      tr_Tbar_Vis.rename("tr_Tbar","")         
      
      Tbar0   = Tbar - (1/3)*tr(Tbar)*Identity(3)
      #
      Misesbar = sqrt((3/2)*inner(Tbar0, Tbar0))
      Misesbar_Vis = project(Misesbar,W)
      Misesbar_Vis.rename("Misesbar, kPa"," ")    
     
     # Write field quantities of interest
      file_results.write(ubar_Vis, t)
      file_results.write(pbar_Vis, t)
      #
      file_results.write(J_Vis, t)  
      file_results.write(Je_Vis, t)  

      #
      file_results.write(P11bar_Vis, t)  
      file_results.write(P22bar_Vis, t)    
      file_results.write(P33bar_Vis, t)  
      file_results.write(P12bar_Vis, t)  
      file_results.write(P13bar_Vis, t)    
      file_results.write(P23bar_Vis, t)  
      file_results.write(P21bar_Vis, t)  
      file_results.write(P31bar_Vis, t)    
      file_results.write(P32bar_Vis, t)  
      file_results.write(T11bar_Vis, t)  
      file_results.write(T22bar_Vis, t)    
      file_results.write(T33bar_Vis, t)  
      file_results.write(T12bar_Vis, t)  
      file_results.write(T13bar_Vis, t)    
      file_results.write(T23bar_Vis, t)  
      file_results.write(T21bar_Vis, t)  
      file_results.write(T31bar_Vis, t)    
      file_results.write(T32bar_Vis, t)
      file_results.write(tr_Tbar_Vis, t)
      file_results.write(Misesbar_Vis, t)   

      file_results.write(Tsh11bar_Vis, t)  
      file_results.write(Tsh22bar_Vis, t)    
      file_results.write(Tsh33bar_Vis, t)   
     
# Write initial state to XDMF file
writeResults(tdummy)  


print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()

# Give the step a descriptive name
step = "Void Expansion"

if(simulation_type==2):
    Volume_void0 = assemble(pi*pow((x[0]+ubar[0]),2)*(-n[0])*ds(surface_boundary_number)) #will be approx 2*pi*(a0)^2 b0 /3
elif(simulation_type==3):
    area_void0 = assemble((x[0]+ubar[0])*(-n[0])*ds(surface_boundary_number)) #will be approx pi a0 b0 /4

Jg_numerical = 1.0


# Initialize an  array for storing results
siz       = 100000 
storage_var      = np.zeros([siz+1,4])
storage_var[0,0] = Jg_numerical    
storage_var[0,1] = a0/b0 #Initial aspect ratio  
storage_var[0,2] = tdummy
storage_var[0,3] = pbarb_max*tdummy

# Set increment counter to zero
ii = 0
cutbackcounter = 0

while (tdummy <= 1): 
    
    tdummy += float(dt_dummy)
    pbar_b.tdummy = tdummy
    
    try:
        (iter, converged) = solver.solve()
    except:
        print("**************Solver failure***********")
        print("Simulation Time (tdummy): {}  ".format(tdummy))
        print("****************************************")
        converged = False              
         
    # Check if solver converges
    if converged: 
        
        # Increment counter
        ii += 1
        
        # Write output to *.xdmf file
        writeResults(tdummy)
        
        # Update DOFs for next step
        w_old.vector()[:] = w.vector()
        
        # Write time histories     
        #
        a = a0 + w.sub(0).sub(0)(a0, 0) 
        b = b0 + w.sub(0).sub(1)(0, b0) 
        storage_var[ii,1] = a/b 
        storage_var[ii,2] = tdummy        
        storage_var[ii,3] = pbarb_max*tdummy
        
        n_dummy = dot(F2Dcof,n)
        if(simulation_type==2):
            Volume_void = assemble(pi*pow((x[0]+ubar[0]),2)*(-n_dummy[0])*ds(surface_boundary_number))
            Jg_numerical = Volume_void/Volume_void0 
            Jg_numerical_approx = (pow(a,2)*b)/(pow(a0,2)*b0) #Approximate Jg estimate assuming ellipse
        elif(simulation_type==3):
            area_void = assemble((x[0]+ubar[0])*(-n_dummy[0])*ds(surface_boundary_number))  
            Jg_numerical = area_void/area_void0 
            Jg_numerical_approx = a*b/a0*b0 #Approximate Jg estimate assuming ellipse

        storage_var[ii,0] = Jg_numerical  
        if ii%vis_steps == 0 or ii==1:
            print("Jg_num : {}  ".format(Jg_numerical))
            print("Jg_num_approx(assuming ellipse) : {}  ".format(Jg_numerical_approx))
        
        # Print progress of calculation periodically
        if ii%vis_steps == 0 or ii==1:               
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            RunTime = datetime.now() - startTime

            print("Real run time:  {}".format(RunTime))
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time (tdummy): {}  ".format(tdummy))
            print("Aspect ratio a/b : {}".format(a/b))
            print("------------------------------------------")             
          
        
        if(adaptive_solver):
            # Iteration-based adaptive time-stepping
            if iter<=adaptive_solver_iters_min:
                dt = adaptive_solver_increasefactor*dt
                dt_dummy.assign(dt)
                print("Adaptive stepper increasing step size due to low convergence iterations")
                print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
                print("Current dt:  {}".format(dt)) 
            elif iter>=adaptive_solver_iters_max:
                dt = dt/adaptive_solver_decreasefactor
                dt_dummy.assign(dt)    
                cutbackcounter +=1
                print("Adaptive stepper cutting back due to high convergence iterations")
                print("Cutback counter:  {}".format(cutbackcounter)) 
                print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
                print("Current dt:  {}".format(dt)) 

    # If solver doesn't converge, don't save results and try a smaller dt 
    else:  
        tdummy -= float(dt_dummy)
        # Reset DOFs for next step
        w.vector()[:] = w_old.vector()
        
        if(adaptive_solver):
            cutbackcounter +=1
            print("Adaptive stepper cutting back due to failed convergence")
            print("Cutback counter:  {}".format(cutbackcounter)) 
            print("Simulation Time before cutback (tdummy): {}  ".format(tdummy+float(dt_dummy)))
            # cut back on dt
            dt = dt/adaptive_solver_decreasefactor
            dt_dummy.assign(dt)
            print("Current dt:  {}".format(dt))  
            
        else:
            print("Main solver failed during void expansion")  
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time (tdummy): {}  ".format(tdummy))
            break                       

    if(adaptive_solver):    
        if(cutbackcounter>adaptive_solver_cutbacklimit):
            print("Main solver failed during void expansion")  
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time (tdummy): {}  ".format(tdummy))
            break
        
print("-----------------------------------------")
print("Saving data in matlab and pickle files")

# Only save as far as time history data
ind = np.argmax(storage_var[:,2])
mdic = {"storage_var": storage_var[0:ind+1,:]}
savenamemat = savename + ".mat"
savenamepickle = savename + ".pickle"    
scipy.io.savemat(savenamemat, mdic)
with open(savenamepickle, 'wb') as f:
    pickle.dump(storage_var[0:ind+1,:], f)
print("-----------------------------------------")

savenamepng = savename + ".png" 
savenameeps = savename + ".eps" 
        
# End analysis
print("-----------------------------------------")
print("End computation")                 
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------------")

'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''

fig = plt.figure() 
ax=fig.gca()  

# Set up font size, initialize colors array
font = {'size'   : 14}
plt.rc('font', **font)
#
prop_cycle = plt.rcParams['axes.prop_cycle']
colors     = prop_cycle.by_key()['color']

plt.plot( storage_var[0:ind+1,0], storage_var[0:ind+1,1], linewidth=1.0, \
         label= "void growth")


#-----------------------------------------------------
plt.grid(linestyle="--", linewidth=0.5, color='b')
#ax.set_xlabel("Jg_numerical",size=14)
ax.set_xlabel("Volume ratio",size=14)
ax.set_ylabel("Aspect ratio b/a",size=14)
plt.legend()
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())

import matplotlib.ticker as ticker
ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.2f'))

plt.savefig(savenamepng)
plt.savefig(savenameeps, format='eps')

plt.show()
    
    
    
