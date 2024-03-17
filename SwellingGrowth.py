"""
Swelling Growth Code

Problem: Time dependent applied pressure (solution known), and growth against
contact like planes, rings and boxes. Axisymmetric, 3D and plane strain 
implementations all included.

Dimensionless version of theory is implemented in the code

Degrees of freedom:
#
Displacement: ubar
pressure: pbar
chemical potential: mubar
Swelling ratio: Js
Growth Volume ratio: Jg
Right Cauchy Growth shape/distortion tensor: Cgsh (Cg_dis in thesis)

Chockalingam
chocka94@mit.edu
chocka94@gmail.com

Jan 2024
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
# Ellipse major axis
a0 = 1 
# Ellipse minor axis
b0 = 1 

applied_pb_problem = 0

if(applied_pb_problem):
    simulation_type = 2 #1 is full3D, 2 is axisymmetric, 3 is plane strain

fine_mesh = 1 #Set to 1 for fine surface mesh in the 2D body (for contact)

if(applied_pb_problem):
    contact_type = 0
    WallSwitch = 0.0
else:
    WallSwitch = 1.0
    contact_type = 5 #Contact type 1 is x_wall, 2 is axisymmetric r wall, 3 is axisymmetric z wall,
                    #4 is axisymmetric ring, 5 is 2D box, 6 is 3D box

if(contact_type): 
    if(contact_type==1):
        simulation_type = 3 #Choice between 1 and 3
    elif(contact_type>=2 and contact_type<=4):
        simulation_type = 2 #Axisymmetric
    elif(contact_type==5):
        simulation_type =3
    elif(contact_type==6): 
        simulation_type =1

compressibility_model = 1 #1 is p = -K lnJe/(JeJs), 2 is p = -K lnJe/(Je)

# Initialize an empty mesh object
mesh = Mesh()

if(simulation_type==1):
    
    with XDMFFile("meshes/Sphere_Quadrant.xdmf") as infile:
        infile.read(mesh)
    
    # Read the 2D subdomain data stored in the *.xdmf file
    mvc2d = MeshValueCollection("size_t", mesh, 2)
    with XDMFFile("meshes/facet_Sphere_Quadrant.xdmf") as infile:
        infile.read(mvc2d, "name_to_read")
    
    # Mesh facets
    facets = cpp.mesh.MeshFunctionSizet(mesh, mvc2d)
    
    #From gmsh
    #Physical Surface("boundary_surface",1) = {1};
    #Physical Surface("ZZero",2) = {2};
    #Physical Surface("XZero",3) = {3};
    #Physical Surface("YZero",4) = {4};
    surface_boundary_number = 1
    
else: 
    
    if(fine_mesh!=1):
        # Read the .xdmf  file data into mesh object
        with XDMFFile("meshes/Circular_Body.xdmf") as infile:
            infile.read(mesh)
            
        # Read mesh boundary facets.
        # Specify parameters of 1-D mesh function object
        mvc_1d = MeshValueCollection("size_t", mesh, 1)
        # Read the   *.xdmf  data into mesh function object
        with XDMFFile("meshes/facet_Circular_Body.xdmf") as infile:
            infile.read(mvc_1d, "name_to_read")
    else:
        # Read the .xdmf  file data into mesh object
        with XDMFFile("meshes/Circular_Body_FineSurface.xdmf") as infile:
            infile.read(mesh)
            
        # Read mesh boundary facets.
        # Specify parameters of 1-D mesh function object
        mvc_1d = MeshValueCollection("size_t", mesh, 1)
        # Read the   *.xdmf  data into mesh function object
        with XDMFFile("meshes/facet_Circular_Body_FineSurface.xdmf") as infile:
            infile.read(mvc_1d, "name_to_read")
    
    # Store boundary facets data
    facets = cpp.mesh.MeshFunctionSizet(mesh, mvc_1d)
    
    #From gmsh
    #Physical Curve('bottom', 14) 
    #Physical Curve('left', 17) 
    #Physical Curve("boundary_curve", 18) 
    surface_boundary_number = 18
 
# Extract initial mesh coords
x = SpatialCoordinate(mesh)
 
# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)

# Facet normal
if(simulation_type==1):
    n = FacetNormal(ds) 
else:
    n = FacetNormal(mesh)
    
'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
KbyG = Constant(10000)
Gbymustar = Constant(4e-3)
Kbymustar = KbyG*Gbymustar
chi     = Constant(0.6)            # Flory-Huggins mixing parameter
#
phi0    = Constant(0.999)          # Initial polymer volume fraction (does not affect solution)
taugbytaud = Constant(1e12) #Set to large value to neglect diffusion-consumption limitation effects
eta = Constant(0.8) #eta should be < 1
#Reduce eta when Jsfree<Jsc (eta may have to be negative IF Jsfree<Jsc for eta=0)

if (float(chi)>0.5):
    fgbarmix_max = (chi - ln(2*chi))
else :
    fgbarmix_max = 1 - chi
    
Delta_mu0bar = -eta*fgbarmix_max #Dimensionless species conversion cost

#taug_dis/tau_g in thesis
taug_sh_by_taug = 1e-2 #Timescale of shape evolution as a fraction 
                      #of volumetric growth timescale


'''''''''''''''''''''
Functions to determine free swelling ratio, critical swelling ratio and
the columetric growth rate as a function of Js. Assumes near incompressibility
'''''''''''''''''''''

def mumixbar(phi):
    mumixbar = np.log(1.0-phi) + phi + float(chi)*phi*phi 
    return mumixbar

def Eqb_swelling_res(phi,pbbar):
    res = (mumixbar(phi)/float(Gbymustar)) + pow(phi,1./3) - phi + pbbar
    return res

def Eqb_swelling_res2(pbbar,phi): #Different passed variables from above
    res = (mumixbar(phi)/float(Gbymustar)) + pow(phi,1./3) - phi + pbbar
    return res

def Eqb_swelling_res_2D(phi,pbbar): #In case of plane strain simulations
    res = (mumixbar(phi)/float(Gbymustar)) + 1.0 - phi + pbbar
    return res

def Eqb_swelling_res2_2D(pbbar,phi):
    res = (mumixbar(phi)/float(Gbymustar)) + 1.0 - phi + pbbar
    return res

def Eqb_Jeres(Je,phi_eqb):
    res = (mumixbar(phi_eqb)/float(Kbymustar)) - np.log(Je)
    return res

def fgbar_mix(phi):
    fgbarmix = 1 + ln(1-phi) + chi*(2*phi-1) 
    return fgbarmix

def floatfgbar_mix(phi):
    fgbarmix = 1 + np.log(1-phi) + float(chi)*(2*phi-1) 
    return fgbarmix

def Gamma_bar_func(phi,phi_free):
    denom =  Delta_mu0bar +  fgbar_mix((phi_free))
    Gamma_bar = (Delta_mu0bar +  fgbar_mix(phi))/denom
    Gamma_bar = conditional(gt(Gamma_bar,0),Gamma_bar,0.0) #No de-growth
    return Gamma_bar

def Gamma_bar_func2(phi,phi_free):
    denom =  float(Delta_mu0bar) +  floatfgbar_mix(phi_free)
    Gamma_bar = (float(Delta_mu0bar) +  floatfgbar_mix(phi))/denom
    return Gamma_bar

from scipy.optimize import fsolve

#Determine free swelling ratio
if(simulation_type==3): #plane strain
    phi_free = float(fsolve(Eqb_swelling_res_2D, 0.99,0.0))
    Js_free = 1./phi_free
    print("Free swelling ratio 2D")
    print(Js_free) 
else:
    phi_free = float(fsolve(Eqb_swelling_res, 0.99,0.0))
    Js_free = 1./phi_free
    print("3D Free swelling ratio")
    print(Js_free)

phi_c = float(fsolve(Gamma_bar_func2, 0.99,phi_free))
Js_c = 1./phi_c
print("Critical swelling ratio")
print(Js_c)

if(simulation_type==3):
    pbar_b_c = float(fsolve(Eqb_swelling_res2_2D, 0.0,phi_c))
    print("2D Homeostatic pressure pbar_b_c")
    print(pbar_b_c)
else:
    pbar_b_c = float(fsolve(Eqb_swelling_res2, 0.0,phi_c))
    print("3D Homeostatic pressure pbar_b_c")
    print(pbar_b_c)

if(applied_pb_problem):
    #Pressure loading (linear ramping from zero pressure)
    pbar_b_frac = 1.5 #Max applied pressure as fraction of homeostatic value
else:
    pbar_b_frac = 0.0
    
pbar_b0 = pbar_b_frac*pbar_b_c #Max applied pressure
print("Max Applied pressure (pb_bar0):")
print(float(pbar_b0))

if(applied_pb_problem):

    if(simulation_type==3):
        phi_eqb = float(fsolve(Eqb_swelling_res_2D, 0.99,float(pbar_b0)))
    else:
        phi_eqb = float(fsolve(Eqb_swelling_res, 0.99,float(pbar_b0)))
    print("Equilibrium swelling ratio at max pressure")
    print(1/phi_eqb)
    
    print("Expected Gammabar at max pressure")
    Gamma_bar_eqb = Gamma_bar_func(phi_eqb,phi_free)
    print(float(Gamma_bar_eqb))
    
    print("Expected approx Je at max pressure") #Approx since material is not perfectly incompressible
    Je_eqb = float(fsolve(Eqb_Jeres, 0.999,phi_eqb))
    print(Je_eqb)
    
    print("Expected approx Je at free growth")
    Je_free = float(fsolve(Eqb_Jeres, 0.999,phi_free))
    print(Je_free)


"""
Simulation time-control related params
"""
tgbar    = 0.0 
dt = 0.05 # Initial float value of dtgbar (can be changed by adaptive stepper)
tg_f = 5.0
vis_steps = 5 #print detailed results for every vis_steps steps

adaptive_solver = 1 #Adaptive time stepping if value is 1
adaptive_solver_iters_min = 2 #Increasing stepping time if solver iterations below this value
adaptive_solver_iters_max = 15 #Decrease stepping time if solver iterations greater than this value
adaptive_solver_cutbacklimit = 3 #Maximum number of times you want solver to decrease stepping time
adaptive_solver_decreasefactor = 2 #Factor by which you want to cut stepping time
adaptive_solver_increasefactor = 1.5 #Factor by which you want to increase stepping time

storage_plots = 1.0 #set to zero if you dont want plots
GrowthSwitch = 1.0
GrowthShapeSwitch = 1.0

if(contact_type):
    if(simulation_type==3):
        r_free = (Js_free)**0.5
    else:
        r_free = (Js_free)**(1./3.)
    
    #Contact type 1 is x_wall, 2 is axisymmetric r wall, 3 is axisymmetric z wall,
    #4 is axisymmetric ring, 5 is 2D box, 6 is 3D box    
    if(contact_type==1):
        x_wall = 3.0
        if(x_wall <= 1.1*r_free):
            print("Warning: Wall might interfere with free swelling state")
    elif(contact_type==2):
        r_wall = 3.0
        if(r_wall <= 1.1*r_free):
            print("Warning: Wall might interfere with free swelling state")
    elif(contact_type==3):
        z_wall = 3.0
        if(z_wall <= 1.1*r_free):
            print("Warning: Wall might interfere with free swelling state")
    elif(contact_type==4): #Axisymmetric ring
        r_ring = 3.0
        h_ring = 0.5
        if(r_ring <= 1.1*r_free):
            print("Warning: Ring might interfere with free swelling state")
    elif(contact_type==5):
        x_wall = 3.0
        y_wall = x_wall
        if(y_wall <= 1.1*r_free):
            print("Warning: Wall might interfere with free swelling state")    
    elif(contact_type==6):
        x_wall = 3.0
        y_wall = x_wall
        z_wall = y_wall        
        if(z_wall <= 1.1*r_free):
            print("Warning: Wall might interfere with free swelling state") 

# Boundary condition expression for increasing the applied pressure 
pbar_b = Expression(("pbar_b0*tg_bar/tgf"),
                     pbar_b0 = float(pbar_b0), tg_bar = tgbar, tgf = tg_f, degree=1)

def dummy_pbarfunc_t(tg_bar,tg_f):
    pbar_b =  float(pbar_b0)*tg_bar/tg_f 
    return pbar_b

'''''''''''''''''''''
Function spaces
'''''''''''''''''''''
# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For ubar
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For pbar, mubar, Js, Jg
if(simulation_type==3):
    T1 = TensorElement("Lagrange", mesh.ufl_cell(), 1, shape=(2, 2), symmetry =True) # for Cgsh plane strain
else:
    T1 = TensorElement("Lagrange", mesh.ufl_cell(), 1, shape=(3, 3), symmetry =True) # for Cgsh


#
TH = MixedElement([U2, P1, P1, P1, P1, T1]) # Taylor-Hood style mixed element
ME = FunctionSpace(mesh, TH)    # Total space for all DOFs

# Define trial functions
w = Function(ME)
ubar, pbar, mubar, Js, Jg, Cgsh = split(w)  

# A copy of functions to store values in the previous step
w_old = Function(ME)
ubar_old,   pbar_old,  mubar_old,  Js_old, Jg_old, Cgsh_old  = split(w_old)   

# Define test functions
w_test = TestFunction(ME)                
ubar_test, pbar_test, mubar_test, Js_test, Jg_test, Cgsh_test  = split(w_test)   

# Define trial functions needed for automatic differentiation
dw = TrialFunction(ME)             

# Assign initial value of mubar
mubar0     = mumixbar(float(phi0))
init_mubar = Constant(mubar0) 
mubar_init = interpolate(init_mubar,ME.sub(2).collapse())
assign(w_old.sub(2),mubar_init) 
assign(w.sub(2), mubar_init)

# Assign initial  value of Js
Js0 = 1/phi0 
init_Js = Constant(Js0)
Js_init = interpolate(init_Js, ME.sub(3).collapse())
assign(w_old.sub(3), Js_init)
assign(w.sub(3), Js_init)

# Assign initial value of Jg
init_Jg = Constant(1.0) 
Jg_init = interpolate(init_Jg,ME.sub(4).collapse())
assign(w_old.sub(4),Jg_init) 
assign(w.sub(4), Jg_init)

# Assign initial value of Cgsh
T1_state = FunctionSpace(mesh, T1)
if(simulation_type==3):
    init_Cgsh = project(Identity(2), T1_state) #For plane strain
else:
    init_Cgsh = project(Identity(3), T1_state)
assign(w_old.sub(5),init_Cgsh)
assign(w.sub(5),init_Cgsh)

''''''''''''''''''''''
Subroutines for the weakforms
'''''''''''''''''''''

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
    Je    = (detF/Js/Jg)     
    return   Je    

# Plane strain Cgsh (3D)
def Cgsh_pe_calc(Jg, Cgsh):
   return as_tensor([[pow(Jg,1./3)*Cgsh[0,0], pow(Jg,1./3)*Cgsh[0,1], 0],
                     [pow(Jg,1./3)*Cgsh[1,0], pow(Jg,1./3)*Cgsh[1,1], 0],
                     [0, 0, pow(Jg,-2./3)]])

# Normalized Piola stress 
def Piola_calc(u,pbar,Jg,Cgsh):
    F = F_calc(u)
    J = det(F)
    if(simulation_type==3):
        Cgsh3D =  Cgsh_pe_calc(Jg, Cgsh)
        Tmatbar = Jg*(F*pow(Jg,-2./3)*inv(Cgsh3D) - inv(F.T) ) - J*pbar*inv(F.T)
    else:
        Tmatbar = Jg*(F*pow(Jg,-2./3)*inv(Cgsh) - inv(F.T) ) - J*pbar*inv(F.T)
    return Tmatbar

# Normalized species flux
def Flux_calc(u, mubar):
    F = F_calc(u) 
    J = det(F)
    Cinv = inv(F.T*F) 
    Mobbar = J*Cinv
    Jmatbar = - Mobbar * grad_scalar(mubar)
    return Jmatbar


#Functions for contact penalty conditions:

# Macaulay bracket function
def ppos(x):
    return (x+abs(x))/2.

def heavyside(x):
    heavy_x = conditional(gt(x,0),1,0)
    return heavy_x 

def smoothheavyside(x,k_tanh):
    smoothheavy_x = ppos(tanh(k_tanh*x))
    return smoothheavy_x 

#-----------------------------------------------------------------------------

'''''''''''''''''''''''''''''
Kinematics 
'''''''''''''''''''''''''''''
F = F_calc(ubar)
J = det(F)  # Total volumetric jacobian
C = (F.T)*F

J_Csh = det(Cgsh)

Fcof = J*inv(F.T)

if(simulation_type!=1):
    # Cofactor of F
    F2D = F_calc2D(ubar);
    C2D = (F2D.T)*(F2D)
    J2D = det(F2D)
    F2Dcof = J2D*inv(F2D.T)
    Fcof_2D = as_tensor([[Fcof[0,0], Fcof[0,1]],
              [Fcof[1,0], Fcof[1,1]]]) #DIFFERENT FROM F2dcof for axisym
 
if(simulation_type==3):
    beta = (1./2)*(inner(inv(Cgsh),C2D))*pow(Jg,-1.0) 
else:
    beta = (1./3)*(inner(inv(Cgsh),C))*pow(Jg,-2./3)

if(storage_plots):
    if(simulation_type==2): #2pi factor ignored
        tot_J0 = assemble(x[0]*dx(domain=mesh))
        tot_Jg0 = assemble(x[0]*Jg*dx(domain=mesh))
        tot_Js0 = assemble(x[0]*Js*dx(domain=mesh))
    else:
        tot_J0 = assemble(Constant(1)*dx(domain=mesh))
        tot_Jg0 = assemble(Jg*dx(domain=mesh))
        tot_Js0 = assemble(Js*dx(domain=mesh))    

# Elastic volumetric Jacobian
Je     = Je_calc(ubar,Js,Jg)                    
Je_old = Je_calc(ubar_old,Js_old,Jg_old)

#  Normalized Piola stress
Tmatbar = Piola_calc(ubar, pbar, Jg, Cgsh)

#  Normalized species  flux
Jmatbar = Flux_calc(ubar, mubar)

''''''''''''''''''''''
WEAK FORMS
'''''''''''''''''''''''
# Residuals:
# Res_0: Balance of forces (test fxn: ubar)
# Res_1: Pressure variable (test fxn: pbar)
# Res_2: Balance of mass   (test fxn: mubar)
# Res_3: mu-Js relation (test fxn: Js)
# Res_4: Growth eqn (test fxn: Jg)
# Res_5: Growth shape evolution (text fxn: Cgsh)
# Res_contact: Contact penalty

# Time step field
dtgbar = Constant(dt)
GSwitch = Constant(GrowthSwitch)
WSwitch = Constant(WallSwitch)
GshSwitch = Constant(GrowthShapeSwitch)

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
    if(compressibility_model==1):
        Res_1 = dot((pbar*Je*Js/KbyG + ln(Je)) , pbar_test)*x[0]*dx 
    else:
        Res_1 = dot((pbar*Je/KbyG + ln(Je)) , pbar_test)*x[0]*dx
else:
    if(compressibility_model==1):
        Res_1 = dot((pbar*Je*Js/KbyG + ln(Je)) , pbar_test)*dx
    else:
        Res_1 = dot((pbar*Je/KbyG + ln(Je)) , pbar_test)*dx  

# The weak form for the mass balance
Gammabar =   Gamma_bar_func(1/Js,Constant(phi_free))
if(simulation_type==2):      
    Res_2 =  dot(Js*Jg*Gammabar/taugbytaud, mubar_test)*x[0]*dx \
             -  dot(Jmatbar , grad_scalar(mubar_test) )*x[0]*dx
else:
    Res_2 =  dot(Js*Jg*Gammabar/taugbytaud, mubar_test)*dx \
             -  dot(Jmatbar , grad_scalar(mubar_test) )*dx    

# The weak form for the Js-mu relation
fac = 1/Js
fac2 =  mubar - ( ln(1.0-fac)+ fac + chi*fac*fac) - (Gbymustar)*pbar*Je 

if(simulation_type==2):
     Res_3 = dot(fac2, Js_test)*x[0]*dx   
else:
    Res_3 = dot(fac2, Js_test)*dx

#The weak form for volumetric growth evolution
if(simulation_type==2):   
    Res_4 =  dot((Jg - Jg_old)/dtgbar - Jg*Gammabar*GSwitch, Jg_test)*x[0]*dx        
else:
    Res_4 =  dot((Jg - Jg_old)/dtgbar - Jg*Gammabar*GSwitch, Jg_test)*dx         

#The weak form for the shape evolution
if(simulation_type==2):
    Res_5 = inner((Cgsh - Cgsh_old)/dtgbar, Cgsh_test)*x[0]*dx \
            - GshSwitch*(1./taug_sh_by_taug)*inner((pow(Jg,-2./3)*C - beta*Cgsh),Cgsh_test)*x[0]*dx
elif(simulation_type==1):
    Res_5 = inner((Cgsh - Cgsh_old)/dtgbar, Cgsh_test)*dx \
            - GshSwitch*(1./taug_sh_by_taug)*inner((pow(Jg,-2./3)*C - beta*Cgsh),Cgsh_test)*dx
elif(simulation_type==3): #Plane strain
    Res_5 = inner((Cgsh - Cgsh_old)/dtgbar, Cgsh_test)*dx \
            - GshSwitch*(1./taug_sh_by_taug)*inner((pow(Jg,-1.0)*C2D - beta*Cgsh),Cgsh_test)*dx  

#Contact type 1 is x_wall, 2 is axisymmetric r wall, 3 is axisymmetric z wall,
#4 is axisymmetric ring, 5 is 2D box, 6 is 3D box
k_pen = 1e5 # penalty stiffness
if(contact_type==1):
    f_pen       = -k_pen*ppos(x[0] + ubar[0] - x_wall) # spatial penalty force, scalar
    Res_contact = -dot(f_pen, ubar_test[0])*ds(surface_boundary_number)
elif(contact_type==2):
    f_pen       = -k_pen*ppos(x[0] + ubar[0] - r_wall) # spatial penalty force, scalar
    Res_contact = -dot(f_pen, ubar_test[0])*x[0]*ds(surface_boundary_number)
elif(contact_type==3):
    f_pen       = -k_pen*ppos(x[1] + ubar[1] - z_wall) # spatial penalty force, scalar
    Res_contact = -dot(f_pen, ubar_test[1])*x[0]*ds(surface_boundary_number)    
elif(contact_type==4):
    k_tanh = Constant(100.0)
    f_pen       = -k_pen*ppos(  x[0] + ubar[0] - r_ring)*\
                  (smoothheavyside((h_ring - x[1] -  ubar[1]),k_tanh))
    Res_contact = -dot(f_pen, ubar_test[0])*x[0]*ds(surface_boundary_number) 
elif(contact_type==5):
    f_pen       = -k_pen*ppos(  x[0] + ubar[0] - x_wall ) # spatial penalty force, scalar
    f_pen2      = -k_pen*ppos(  x[1] + ubar[1] - y_wall ) 
    Res_contact1 = -dot(f_pen, ubar_test[0])*ds(surface_boundary_number)  
    Res_contact2 = -dot(f_pen2, ubar_test[1])*ds(surface_boundary_number)
    Res_contact = Res_contact1 + Res_contact2 
elif(contact_type==6):    
    f_pen       = -k_pen*ppos(  x[0] + ubar[0] - x_wall ) # spatial penalty force, scalar
    f_pen2      = -k_pen*ppos(  x[1] + ubar[1] - y_wall ) 
    f_pen3      = -k_pen*ppos(  x[2] + ubar[2] - z_wall )
    Res_contact1 = -dot(f_pen, ubar_test[0])*ds(surface_boundary_number)  
    Res_contact2 = -dot(f_pen2, ubar_test[1])*ds(surface_boundary_number)
    Res_contact3 = -dot(f_pen3, ubar_test[2])*ds(surface_boundary_number)
    Res_contact = Res_contact1 + Res_contact2 + Res_contact3
 
if(contact_type==0):
    Res = Res_0 + Res_1 + Res_2 + Res_3 + Res_4 + Res_5
else:    
    Res = Res_0 + Res_1 + Res_2 + Res_3 + Res_4 + Res_5 + WSwitch*Res_contact

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

'''''''''''''''''''''''
Dirichlet  Boundary Conditions
'''''''''''''''''''''''      
if(simulation_type==1):
    # Gmsh labels
    #Physical Surface("ZZero",2) = {2};
    #Physical Surface("XZero",3) = {3};
    #Physical Surface("YZero",4) = {4};
    bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 3)  # u1 fix  
    bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 4)  # u2 fix 
    bcs_3 = DirichletBC(ME.sub(0).sub(2), 0, facets, 2)  # u3 fix 
    bcs_4 = DirichletBC(ME.sub(2), 0, facets, 1)   # chem. pot. Boundary
    # 
    bcs = [bcs_1, bcs_2, bcs_3, bcs_4]
else:
    # Gmsh labels
    #Physical Curve('bottom', 14) 
    #Physical Curve('left', 17) 
    #Physical Curve("boundary_curve", 18) 
    bcs_1 = DirichletBC(ME.sub(0).sub(0), 0, facets, 17)  # u1 fix - left  
    bcs_2 = DirichletBC(ME.sub(0).sub(1), 0, facets, 14)  # u2 fix - bottom
    bcs_3 = DirichletBC(ME.sub(2), 0, facets, 18)   # chem. pot. Boundary
    # 
    bcs = [bcs_1, bcs_2, bcs_3]

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
if(applied_pb_problem):
    savename = ("chi{0:.1f}_eta{1:.3f}_Gbymustar{2:.2e}_pbfrac{3:.2f}_tgf{4:.1f}" \
                .format(float(chi),float(eta),float(Gbymustar),float(pbar_b_frac),float(tg_f)))
    if(simulation_type==2):    
        savename = "results/TimedepPressure/Axisym"+ savename    
    elif(simulation_type==1):
        savename = "results/TimedepPressure/3D"+ savename
    elif(simulation_type==3):
        savename = "results/TimedepPressure/PlaneStrain"+ savename    
elif(contact_type==4):
    savename = ("results/Ring_chi{0:.1f}_eta{1:.3f}_Gbymustar{2:.2e}/taugsh{3:.1e}_kpen{4:.1e}_cback{5:d}_tgf{6:.1f}_adapminiter{7:d}" \
                .format(float(chi),float(eta),float(Gbymustar),float(taug_sh_by_taug),float(k_pen), \
                        adaptive_solver_cutbacklimit,float(tg_f),adaptive_solver_iters_min))      
elif(contact_type==3):
    savename = ("results/AxisymZwall_chi{0:.1f}_eta{1:.3f}_Gbymustar{2:.2e}/taugsh{3:.1e}_kpen{4:.1e}_cback{5:d}_tgf{6:.1f}_adapminiter{7:d}" \
                .format(float(chi),float(eta),float(Gbymustar),float(taug_sh_by_taug),float(k_pen), \
                        adaptive_solver_cutbacklimit,float(tg_f),adaptive_solver_iters_min))   
elif(contact_type==5):
    savename = ("results/2DBox_chi{0:.1f}_eta{1:.3f}_Gbymustar{2:.2e}/taugsh{3:.1e}_kpen{4:.1e}_cback{5:d}_tgf{6:.1f}_adapminiter{7:d}" \
                .format(float(chi),float(eta),float(Gbymustar),float(taug_sh_by_taug),float(k_pen), \
                        adaptive_solver_cutbacklimit,float(tg_f),adaptive_solver_iters_min))   
elif(contact_type==6):
    savename = ("results/3DBox_chi{0:.1f}_eta{1:.3f}_Gbymustar{2:.2e}/taugsh{3:.1e}_kpen{4:.1e}_cback{5:d}_tgf{6:.1f}_adapminiter{7:d}" \
                .format(float(chi),float(eta),float(Gbymustar),float(taug_sh_by_taug),float(k_pen), \
                        adaptive_solver_cutbacklimit,float(tg_f),adaptive_solver_iters_min))  

if(fine_mesh==1):
    savename = savename + "_FINE"  

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
      
      # Visualize pbar
      pbar_Vis = project(pbar, W)
      pbar_Vis.rename("pbar"," ")
      
      # Visualize  mubar
      mubar_Vis = project(mubar, W)
      mubar_Vis.rename("mubar"," ")
      
      # Visualize  Js
      Js_Vis = project(Js, W)
      Js_Vis.rename("Js"," ")
      
      # Visualize  Jg
      Jg_Vis = project(Jg, W)
      Jg_Vis.rename("Jg"," ")
      
      #Vizualize volumetric growth rate
      Gammabar =   Gamma_bar_func(1/Js,Constant(phi_free))
      Gammabar_Vis = project(Gammabar, W)
      Gammabar_Vis.rename("Gammabar"," ")
      
      # Visualize phi
      phi = 1/Js
      phi_Vis = project(phi, W)
      phi_Vis.rename("phi"," ")
      
      # Visualize J
      J_Vis = project(J, W)
      J_Vis.rename("J"," ")    
      
      # Visualize Je
      Je_Vis = project(Je, W)
      Je_Vis.rename("Je"," ")    
      
      JCsh_Vis = project(J_Csh, W)
      JCsh_Vis.rename("JCsh"," ")  
      
      # Visualize Piola stress
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
      
      #T21bar_Vis = project(Tbar[1,0],W)
      #T21bar_Vis.rename("T21bar","")
      #T31bar_Vis = project(Tbar[2,0],W)
      #T31bar_Vis.rename("T31bar","")    
      #T32bar_Vis = project(Tbar[2,1],W)
      #T32bar_Vis.rename("T23bar","")   
      
      tr_Tbar_Vis = project(tr(Tbar),W)
      tr_Tbar_Vis.rename("tr_Tbar","")         
      
      Tbar0   = Tbar - (1/3)*tr(Tbar)*Identity(3)
      devT11bar_Vis = project(Tbar0[0,0],W)
      devT11bar_Vis.rename("devT11bar","")
      devT22bar_Vis = project(Tbar0[1,1],W)
      devT22bar_Vis.rename("devT22bar","")    
      devT33bar_Vis = project(Tbar0[2,2],W)
      devT33bar_Vis.rename("devT33bar","")  
      
      devT12bar_Vis = project(Tbar0[0,1],W)
      devT12bar_Vis.rename("devT12bar","")
      devT13bar_Vis = project(Tbar0[0,2],W)
      devT13bar_Vis.rename("devT13bar","")    
      devT23bar_Vis = project(Tbar0[1,2],W)
      devT23bar_Vis.rename("devT23bar","")        
  
      
      tr_Tbar_Vis = project(tr(Tbar),W)
      tr_Tbar_Vis.rename("tr_Tbar","")  
      #
      Misesbar = sqrt((3/2)*inner(Tbar0, Tbar0))
      Tbareff = sqrt((3/2)*inner(Tbar, Tbar))
      Misesbar_Vis = project(Misesbar,W)   
      Misesbar_Vis.rename("Misesbar"," ")    
      Tbareff_Vis = project(Tbareff,W)   
      Tbareff_Vis.rename("Tbar_eff"," ")  


      Cgsh_11_Vis = project(Cgsh[0,0],W)
      Cgsh_11_Vis.rename("Cgsh_11"," ")
      Cgsh_22_Vis = project(Cgsh[1,1],W)
      Cgsh_22_Vis.rename("Cgsh_22"," ")
      if(simulation_type!=3):
          Cgsh_33_Vis = project(Cgsh[2,2],W)
          Cgsh_33_Vis.rename("Cgsh_33"," ")
      
      Cgsh_12_Vis = project(Cgsh[0,1],W)
      Cgsh_12_Vis.rename("Cgsh12","")
      if(simulation_type!=3):
          Cgsh_13_Vis = project(Cgsh[0,2],W)
          Cgsh_13_Vis.rename("Cgsh13","")    
          Cgsh_23_Vis = project(Cgsh[1,2],W)
          Cgsh_23_Vis.rename("Cgsh23","") 
     
     # Write field quantities of interest
      file_results.write(ubar_Vis, t)
      file_results.write(pbar_Vis, t)
      file_results.write(mubar_Vis, t)
      file_results.write(Js_Vis, t)
      file_results.write(phi_Vis, t)
      #
      file_results.write(J_Vis, t)  
      file_results.write(Je_Vis, t)  
      file_results.write(Jg_Vis, t) 
      file_results.write(JCsh_Vis, t)
      file_results.write(Gammabar_Vis, t) 
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
      #file_results.write(T23bar_Vis, t)  
      #file_results.write(T21bar_Vis, t)  
      #file_results.write(T31bar_Vis, t)    
      #file_results.write(T32bar_Vis, t)
      file_results.write(devT11bar_Vis, t)  
      file_results.write(devT22bar_Vis, t)    
      file_results.write(devT33bar_Vis, t)  
      file_results.write(devT12bar_Vis, t)  
      file_results.write(devT13bar_Vis, t)    
      file_results.write(devT23bar_Vis, t)  
      file_results.write(tr_Tbar_Vis, t)
      file_results.write(Misesbar_Vis, t) 
      file_results.write(Tbareff_Vis, t) 
      file_results.write(Cgsh_11_Vis, t)
      file_results.write(Cgsh_22_Vis, t) 
      file_results.write(Cgsh_12_Vis, t)
      if(simulation_type!=3):
          file_results.write(Cgsh_33_Vis, t)              
          file_results.write(Cgsh_13_Vis, t) 
          file_results.write(Cgsh_23_Vis, t)    
     
# Write initial state to XDMF file
writeResults(tgbar -1) #Dummy negative time for undeformed state  


def volume_calculations():
    if(simulation_type==2):
        tot_Jgvolume = assemble(Jg*x[0]*dx(domain=mesh))
        tot_Jevolume = assemble(Je*x[0]*dx(domain=mesh))
        tot_Jvolume = assemble(J*x[0]*dx(domain=mesh))
        tot_Jsvolume = assemble(Js*x[0]*dx(domain=mesh))
        tot_Gammabar = assemble(Gammabar*x[0]*dx(domain=mesh))
    else:     
        tot_Jgvolume = assemble(Jg*dx(domain=mesh))
        tot_Jevolume = assemble(Je*dx(domain=mesh))
        tot_Jvolume = assemble(J*dx(domain=mesh))
        tot_Jsvolume = assemble(Js*dx(domain=mesh))
        tot_Gammabar = assemble(Gammabar*dx(domain=mesh))      
    return tot_Jgvolume, tot_Jevolume, tot_Jvolume, tot_Jsvolume, tot_Gammabar
    

def progress_printfunc(tgbar):
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    RunTime = datetime.now() - startTime
    Js_avg = tot_Jsvolume/tot_J0;
    dummy_pbbar = dummy_pbarfunc_t(tgbar,tg_f)
    if(applied_pb_problem):
        if(simulation_type==3):
            phi_eqb = float(fsolve(Eqb_swelling_res_2D, 0.99,float(dummy_pbbar)))
        else:
            phi_eqb = float(fsolve(Eqb_swelling_res, 0.99,float(dummy_pbbar)))
        Gamma_bar_eqb = float(Gamma_bar_func(phi_eqb,phi_free))
        Gamma_bar_eqb2 = float(Gamma_bar_func(1/Js_avg,phi_free))
        Je_eqb = float(fsolve(Eqb_Jeres, 0.999,phi_eqb))
        Je_eqb2 = float(fsolve(Eqb_Jeres, 0.999,1/Js_avg))

    print("Real run time:  {}".format(RunTime))
    print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
    print("Simulation Time (tgbar): {}  ".format(tgbar))
    if(applied_pb_problem):
        print("Current pb_bar : {}".format(dummy_pbbar))
    print("Total volume ratio: {}".format(tot_Jvolume/tot_J0))
    print("Average Growth volume ratio: {}".format(tot_Jgvolume/tot_J0))
    print("Average Swelling volume ratio: {}".format(Js_avg))
    if(applied_pb_problem):
        print("Expected Js: {}".format(1/phi_eqb))
    print("Average Gamma Bar : {}".format(tot_Gammabar/tot_J0))
    if(applied_pb_problem):
        print("Expected Gamma bar: {}".format(Gamma_bar_eqb))
        print("Expected Gamma bar using Js_avg: {}".format(Gamma_bar_eqb2))
    print("Average Je: {}".format(tot_Jevolume/tot_J0))
    if(applied_pb_problem):
        print("Expected Je: {}".format(Je_eqb))
        print("Expected Je using Js_avg: {}".format(Je_eqb2))
    volumeratiocheck = (tot_Jgvolume*tot_Jsvolume*tot_Jevolume/ \
                        pow(tot_J0,3))/(tot_Jvolume/tot_J0);
    print("Ratio check (should be 1): {}".format(volumeratiocheck))
    print("------------------------------------------") 
    
def store_variables(ii):
     storage_var[ii,0] = tgbar # store the time 
     storage_var[ii,1] = tot_Jvolume/tot_J0 #Volume ratio
     storage_var[ii,2] = tot_Jgvolume/tot_J0 #Average Growth
     storage_var[ii,3] =  tot_Jsvolume/tot_J0 #Average Swelling  
     storage_var[ii,4] =  tot_Jevolume/tot_J0 #Average Je 
     
     if(applied_pb_problem):
        dummy_pbbar = dummy_pbarfunc_t(tgbar,tg_f)
        if(simulation_type==3):
             phi_eqb = float(fsolve(Eqb_swelling_res_2D, 0.99,float(dummy_pbbar)))
        else:
             phi_eqb = float(fsolve(Eqb_swelling_res, 0.99,float(dummy_pbbar)))
         
        storage_var[ii,5] = 1/phi_eqb#Volume ratio



print("------------------------------------")
print("Simulation Start")
print("------------------------------------")
# Store start time 
startTime = datetime.now()


print("------------------------------------")
print("Initial swelling equilibrium")
print("------------------------------------")
GrowthSwitch = 0.0
GrowthShapeSwitch = 0.0
GSwitch.assign(GrowthSwitch)
GshSwitch.assign(GrowthShapeSwitch)

step = "SwellingEquilibrium"

(iter, converged) = solver.solve()
writeResults(tgbar)

# Update DOFs for next step
w_old.vector()[:] = w.vector()


if(storage_plots):
    # Initialize an  array for storing results
    ii =0
    siz       = 100000 
    if(applied_pb_problem):
        storage_var = np.zeros([siz+1,6])
    else:
        storage_var = np.zeros([siz+1,5])
    
    tot_Jgvolume, tot_Jevolume, tot_Jvolume, tot_Jsvolume, tot_Gammabar = volume_calculations()
    store_variables(ii)

progress_printfunc(tgbar)


print("------------------------------------")
print("Time evolution simulation starting")
print("------------------------------------")

GrowthSwitch = 1.0
GrowthShapeSwitch = 1.0
GSwitch.assign(GrowthSwitch)
GshSwitch.assign(GrowthShapeSwitch)

# Give the step a descriptive name
step = "SwellingGrowth"

if(adaptive_solver):
    cutbackcounter = 0

while (tgbar <= tg_f):
    
    tgbar += float(dtgbar)
    if(applied_pb_problem):
        pbar_b.tg_bar = tgbar;
    
    try:
        (iter, converged) = solver.solve()
    except:
        print("**************Solver failure***********")
        print("Simulation Time (tgbar): {}  ".format(tgbar))
        print("****************************************")
        converged = False            
         
    # Check if solver converges
    if converged: 
        
        # Increment counter
        ii += 1
        
        tot_Jgvolume, tot_Jevolume, tot_Jvolume, tot_Jsvolume, tot_Gammabar = volume_calculations()
        
        # Write output to *.xdmf file
        writeResults(tgbar)
        
        # Update DOFs for next step
        w_old.vector()[:] = w.vector()
        
        # Write time histories     
        #
        store_variables(ii)
        
        # Print progress of calculation periodically
        if ii%vis_steps == 0 or ii==1:               
            progress_printfunc(tgbar)
            
        if(adaptive_solver):                    
            # Iteration-based adaptive time-stepping
            if iter<=adaptive_solver_iters_min:
                dt = adaptive_solver_increasefactor*dt
                dtgbar.assign(dt)
                print("Adaptive stepper increasing step size due to low convergence iterations")
                print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
                print("Current dt:  {}".format(dt)) 
            elif iter>=adaptive_solver_iters_max:
                dt = dt/adaptive_solver_decreasefactor
                dtgbar.assign(dt)    
                cutbackcounter +=1
                print("Adaptive stepper cutting back due to high convergence iterations")
                print("Cutback counter:  {}".format(cutbackcounter)) 
                print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
                print("Current dt:  {}".format(dt)) 

    else: 
        tgbar -= float(dtgbar)
        w.vector()[:] = w_old.vector()
        if(adaptive_solver):
            dt = dt/adaptive_solver_decreasefactor
            dtgbar.assign(dt)
            cutbackcounter +=1
            print("Adaptive stepper cutting back due to failed convergence")
            print("Cutback counter:  {}".format(cutbackcounter)) 
            print("Current dt:  {}".format(dt))  

        else:
            print("Main solver failed during growth")  
            print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
            print("Simulation Time (tgbar): {}  ".format(tgbar))
            break
    
    if(adaptive_solver):
        if(cutbackcounter>adaptive_solver_cutbacklimit):
                print("Main solver failed during growth")  
                print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
                print("Simulation Time (tgbar): {}  ".format(tgbar))
                break    

if(contact_type):
    
    #Release wall constraint
    print("RELEASING WALL")
    set_log_level(30)
    GrowthSwitch = 0.0
    GrowthShapeSwitch = 0.0
    GSwitch.assign(GrowthSwitch)
    GshSwitch.assign(GrowthShapeSwitch)
    
    dummytime = 0.0
    dt_dummytime = 0.05
     
    #WSwitchcounter = 0
    WallSwitch_n = 1
    
    WallSwitchmin = 1e-4
    WallSwitch_decreasefactor = 2
    
    WallSwitch_old = WallSwitch
    
    if(contact_type==5 or contact_type==6):
        WallSwitchmin = 1e-8
     
    while(WallSwitch>WallSwitchmin):
     
        try:
            (iter, converged) = solver.solve()
        except:
            print("Wall Release solver failed, increasing WSwitch")
            #WallSwitch = WallSwitch*1.5
            WallSwitch = WallSwitch_old
            WallSwitch_n = WallSwitch_n + 1
            WallSwitch = WallSwitch*(WallSwitch_n)/(WallSwitch_n + 1)
            print("Dummy time:  {}".format(dummytime))
            print("WallSwitch value: {}".format(WallSwitch))
            #WSwitchcounter +=1
            if(WallSwitch>1):
                print("ERROR, Solver failed in releasing wall process")
                break
            WSwitch.assign(WallSwitch)
            # Reset DOFs for next step
            w.vector()[:] = w_old.vector()  
            
        if converged:
            writeResults(tgbar+100+dummytime)
            dummytime += dt_dummytime    
            #WallSwitch = WallSwitch/2
            WallSwitch_old = WallSwitch
            WallSwitch = WallSwitch*(WallSwitch_n)/(WallSwitch_n + 1)
            WSwitch.assign(WallSwitch)
            #WSwitchcounter +=1
            print("Release solver converged, decreasing WSwitch")
            print("Dummy time:  {}".format(dummytime))
            print("WallSwitch value: {}".format(WallSwitch))
            w_old.vector()[:] = w.vector()
            
        else:   
            print("Wall Release convergence failed, increasing WSwitch")
            #WallSwitch = WallSwitch*1.5
            WallSwitch = WallSwitch_old
            WallSwitch_n = WallSwitch_n + 1
            WallSwitch = WallSwitch*(WallSwitch_n)/(WallSwitch_n + 1)
            print("Dummy time:  {}".format(dummytime))
            print("WallSwitch value: {}".format(WallSwitch))
            #WSwitchcounter +=1
            #if(WallSwitch>1):
            #    print("ERROR, Solver failed in releasing wall process")
            #    break
            WSwitch.assign(WallSwitch)        
            # Reset DOFs for next step
            w.vector()[:] = w_old.vector()
            
        #if(WSwitchcounter>100):
        if(WallSwitch_n>90):
            print("Wall Release solver failed due to too many iterations")  
            #print("WSwitch counter:  {}".format(WSwitchcounter))
            print("WSwitch n:  {}".format(WallSwitch_n))
            break 
    
    WallSwitch = 0.0
    WSwitch.assign(WallSwitch)
    print("Dummy time:  {}".format(dummytime))
    print("WallSwitch value: {}".format(WallSwitch))
    try:
            (iter, converged) = solver.solve()
    except:
            print("*******Wall Release solver failed for WSwitch = 0*****")
            print("****************************************")
            w.vector()[:] = w_old.vector() 
            converged = False
            
    if converged:        
        writeResults(tgbar+100+dummytime)
        w_old.vector()[:] = w.vector()

                   
# End analysis
print("-----------------------------------------")
print("End time evolution computation")                 
# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("------------------------------------------")

'''''''''''''''''''''
    VISUALIZATION & STORAGE
'''''''''''''''''''''

if(storage_plots):
    
    print("-----------------------------------------")
    print("Saving data in matlab and pickle files")
   
    # Only save as far as time history data
    ind = np.argmax(storage_var[:,0])
    mdic = {"storage_var": storage_var[0:ind+1,:]}
    savenamemat = savename + ".mat"
    savenamepickle = savename + ".pickle"    
    scipy.io.savemat(savenamemat, mdic)
    with open(savenamepickle, 'wb') as f:
       pickle.dump(storage_var[0:ind+1,:], f)
    print("-----------------------------------------")
    
    savenamepng = savename + ".png" 
    savenameeps = savename + ".eps" 
    
    # Set up font size, initialize colors array
    font = {'size'   : 14}
    plt.rc('font', **font)
    #
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors     = prop_cycle.by_key()['color']
       
    #
    fig = plt.figure() 
    ax=fig.gca()  
    plt.plot( storage_var[0:ind,0], storage_var[0:ind,1], 'b', linewidth=1.0,  label= "Volume ratio")
    plt.plot( storage_var[0:ind,0], storage_var[0:ind,2], 'k', linewidth=1.0,  label= "Growth ratio")
    plt.plot( storage_var[0:ind,0], storage_var[0:ind,3], 'r', linewidth=1.0,  label= "Swelling ratio")
    plt.plot(storage_var[0:ind,0], storage_var[0:ind,4], 'g', linewidth=1.0,  label= "Je")
    if(applied_pb_problem):
        plt.plot(storage_var[0:ind,0], storage_var[0:ind,5], '--k', linewidth=1.0,  label= "Expected Js")
    #-----------------------------------------------------
    plt.grid(linestyle="--", linewidth=0.5, color='b')
    ax.set_xlabel("tgbar",size=14)
    ax.set_ylabel("Volume ratios",size=14)
    ax.set_title("Applied time dependent pressure", size=14, weight='normal')
    plt.legend()
    #
    from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.3f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
    
    plt.savefig(savenamepng)
    plt.savefig(savenameeps, format='eps')
    plt.show()
    




