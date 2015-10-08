#########################################################
#     Petr Vagner -- Master thesis
##    SOFC --  simulation of anode supported fuel cell
##    01/07/2014
#########################################################
#
#     cathode means oxygen electrode
#     anode means hydrogen electrode
#
#########################################################
import matplotlib
matplotlib.use('PS')
# used tools
from matplotlib import pyplot as plt
import numpy as np
import time
import datetime
import os
import math
import shutil
from dolfin import *
from os import system, remove
import sys
import subprocess
#
#########################################################
# Generate path with time stamp
#########################################################
temp_dif = sys.argv[1] # T_L - T_0 i.e. 1K 10K
typ    = sys.argv[2] # name_of_the_computation
l_b    = sys.argv[3] # lower bound of phie 0.5 V
u_b    = sys.argv[4] # upper bound of phia 0.6 V
krok   = sys.argv[5] # step between the lower and upper bound f.e. 0.1 V
#rozsah   = sys.argv[6] 
print "Temp dif ",float(temp_dif),"K \n Lower bound ",float(l_b),"V \n Upper bound ", float(u_b)," V\n step ",float(krok),"V"

today = datetime.date.today()
now = datetime.datetime.now()

typ = 'void' # for computation denoting 
#path = "results/"+today.strftime("%d-%m")+"/"+now.strftime("%H-%M-%S")+typ+"/" # results saving path : ~/results/<day-month>/<time+type>/
path =  "results_05K_08/"
#debugging features
save = True
eq_solve = True 
#set_log_level(DEBUG)
parameters['form_compiler']['representation'] = 'uflacs' # magic line by Blechta

def extract_data(mesh, func):
   x = mesh.geometry().x()
   y = np.array([func(Point(s)) for s in x])
   return [x, y]

#########################################################
#  Mesh
#########################################################
mesh = Mesh('scia_3041.xml')
#mesh.coordinates().sort(0)

#########################################################
# Boundaries and subdomains
#########################################################
# thickness of electrolyte is 1.0 implicitly
anode_thickness = 14.0 # thickness of anode according to mesh
cathode_thickness = 4.0 # thickness of cathode acc. to mesh

class Electrolyte(SubDomain):
  def inside(self, x, on_boundary):
    return between(x[0], (0.0, 1.0))

class Anode(SubDomain):
  def inside(self, x, on_boundary):
    return between(x[0], (-anode_thickness, 0.0))

class Cathode(SubDomain):
  def inside(self, x, on_boundary):
    return between(x[0], (1.0, 1.0+cathode_thickness))

class Hydrogen_inlet(SubDomain):
  def inside(self, x, on_boundary):
    tol=1E-07
    return on_boundary and (abs(x[0] + anode_thickness) < tol)

class Oxygen_inlet(SubDomain):
  def inside(self, x, on_boundary):
    tol=1E-07
    return on_boundary and (abs(x[0]-(1+cathode_thickness)) < tol)


## characteric functions of subdomains
class charac_a(Expression):
   def eval(self, value, x ):
      if x[0] < 0.0 - DOLFIN_EPS:
         value[0] = 1.0
      else:
         value[0] = 0.0
class charac_c(Expression):
   def eval(self, value, x ):
      if x[0] > 1.0:
         value[0] = 1.0
      else:
         value[0] = 0.0
def crop(x):
    if abs(x)>1.0e6:
        return 0
    else:
        return x

char_a = charac_a()
char_c = charac_c()
char_e = 1.0 - char_a - char_c
# Initialize sub-domain instances
h_inlet         = Hydrogen_inlet()
anode        = Anode()
electrolyte    = Electrolyte()
cathode         = Cathode()
o_inlet         = Oxygen_inlet()

## Marking subdomains and definition of subdomain measure
dd = CellFunction('size_t', mesh)
dd.set_all(0)
anode.mark(dd, 1)
electrolyte.mark(dd, 2)
cathode.mark(dd, 3)
dx_subdomain = Measure('dx')[dd]

mesh_cathode            = SubMesh(mesh, cathode)
mesh_electrolyte        = SubMesh(mesh, electrolyte)
mesh_anode              = SubMesh(mesh, anode)

## Mark boundary and define surface measure
ff = FacetFunction('size_t', mesh)
ff.set_all(0)
h_inlet.mark(ff,4)
o_inlet.mark(ff,5)
n = FacetNormal(mesh) # snad bude fungovat
ds = Measure('ds')[ff]

###############################################################################################
##
##   Function spaces, functions, test functions
##
###############################################################################################


# Define function spaces and functions
T_space        = FunctionSpace(mesh, "CG", 1) # temperature
I_phi_space    = FunctionSpace(mesh, "CG", 1) # ion potential
E_phi_space    = FunctionSpace(mesh, "CG", 1) # electric potential
O_space         = FunctionSpace(mesh, "CG", 1) # oxygen
H_space         = FunctionSpace(mesh, "CG", 1) # hydrogen
W_space         = FunctionSpace(mesh, "CG", 1) # water vapor

DG_space       = FunctionSpace(mesh, "DG", 1)       # space for projection of discontinuous functions

anode_space             = FunctionSpace(mesh_anode, "CG", 1) 
electrolyte_space       = FunctionSpace(mesh_electrolyte, "CG", 1)
cathode_space           = FunctionSpace(mesh_cathode, "CG", 1)

# joint space for simulatneous solutiuon
V          = MixedFunctionSpace([T_space, I_phi_space, E_phi_space, O_space, H_space, W_space ])

# functions for storing a solution, test functions
v = Function(V)
T, i_phi, e_phi, p_o, p_h, p_w = split(v) # solution
(a, b, c, d, e, f) = TestFunctions(V) # test function

## a -- test function for T    -- temperature
## b -- test function for i_phi   -- ionic potential
## c -- test function for e_phi   -- electrone potential
## d -- test function for p_o  -- oxygen pressure
## e -- test function for p_h  -- parital hydrogen pressure
## f -- test function for p_w  -- parital water vapor pressure

###############################################################################################
##
##    Parameters
##
###############################################################################################

# Constants
L = Constant(1.0e-04) # scaling

## general physics
F = Constant(9.64853399e04) # Faradays constant
R = Constant(8.3144621) #  molar gas constant

## hahahahaha
S_i = Constant(0.0) # entropy of ion in YSZ

#heat conductivity
Lambda_a    = Constant(11.0) # anode
Lambda_e    = Constant(2.0)   # electrolyte
Lambda_c    = Constant(6.0) # cathode

#material characteritics of porous anode and cathode
por_a  = Constant(0.3)      # porosity anode
por_c  = Constant(0.3)      # porosity anode

tur_a  = Constant(1.7) # turtuosity
tur_c  = Constant(1.7) # turtuosity

active_surf_a = Constant(1.174e04)
active_surf_c = Constant(1.174e04)

vol_fr_a    = Constant(0.3)               # Volume fraction of electron conductive material
vol_fr_c    = Constant(0.3)               # Volume fraction of electron conductive material

# conductivity for all materials
## bulk
sigma_i_bulk     = 6.068e03*exp(-8.079e03*inv(T))           # ionic conductivity YSZ - Fischer = 3.25 S/m at 1073K
sigma_LSM_bulk   = Constant(8.855e07)*inv(T)*exp(1082.5*inv(T))      # electric conductivity in cathode electric conductivity LSM = 8.8e07 S/m at 1073 K
sigma_ni_bulk    = 3.27e06 - 1.0653e03*T              # electric conductivity nickel = 2.1e06 at 1073K
## conductivity of porous materials
sigma_lsm      = char_c*por_c*inv(tur_c)*(1.0 - vol_fr_c)*sigma_LSM_bulk
sigma_ni       = char_a*por_a*inv(tur_a)*(1.0 - vol_fr_a)*sigma_ni_bulk
sigma_i_a      = char_a*por_a*inv(tur_a)*vol_fr_a*sigma_i_bulk       # YSZ at anode
sigma_i_c      = char_c*por_c*inv(tur_c)*vol_fr_c*sigma_i_bulk       # YSZ at cathode

# diff coef anode
## molar masses
molar_mass_h     = Constant(2.0e-03)     # hydrogen m.m.
molar_mass_w     = Constant(18.0e-03)    # water vapour m.m.
molar_mass_ox    = Constant(32.0e-03)    # oxygen molar mass 

dif_vol_h     = Constant(6.12e-06)    # diffusion volume of h2 -- Kong 
dif_vol_w     = Constant(1.31e-05)    # diffusion volume of h2 -- Kong
pore_radius   = Constant(2.5e-07)     # average pore radius -- Fischer


kmu = Constant(7.142857142857143e-10) # k/mu for diffusion

D_k_o   = 2*por_c*inv(3*tur_c)*pore_radius*sqrt(8*T*R*inv(math.pi*molar_mass_ox)) # Knudsen coef. for oxygen
D_k_h   = 2*por_a*inv(3*tur_a)*pore_radius*sqrt(8*T*R*inv(math.pi*molar_mass_h))  # Knudsen coef. for hydrogen 
D_k_w   = 2*por_a*inv(3*tur_a)*pore_radius*sqrt(8*T*R*inv(math.pi*molar_mass_w))  # Knudsen coef. for water
## Binary diff coef. for hydrog. and water vap.
D_hw  =   (char_a*por_a*inv(tur_a)*(3.198e-08)*pow(T, 1.75)
    *pow(inv(molar_mass_h)+inv(molar_mass_w), 0.5)
    *inv((p_h+p_w)*pow(pow(dif_vol_h, 0.3333333333)+pow(dif_vol_w,0.3333333333),2)))


### chemical potentials -- fitted from NIST JANAF
# oxygen
mu_o_ref = -(Constant(1.0276975e-09)*(pow(T,4)-pow(300.0,4)) + Constant(-9.673333333333333e-06)*(pow(T,3)-pow(300.0,3)) + Constant(0.04181925)*(pow(T,2)-pow(300.0,2)) + Constant(184.712)*(T-300.0)) 
# hydrogen
mu_h_ref = -(Constant(1.0906075e-09)*(pow(T,4)-pow(300.0,4)) + Constant(-9.820466666666667e-06)*(pow(T,3)-pow(300.0,3)) + Constant(0.0400579)*(pow(T,2)-pow(300.0,2)) + Constant(111.135)*(T-300.0)) 
# water
mu_w_ref = Constant(-228.582e03)  -(Constant(1.087365e-09)*(pow(T,4)-pow(300.0,4)) + Constant(-1.0228033333333333e-05)*(pow(T,3)-pow(300.0,3)) + Constant(0.0468493)*(pow(T,2)-pow(300.0,2)) + Constant(165.31)*(T-300.0))
# energy of formation of water
#dg_w = Constant(1.0e03)*(Constant(-9.33436e-10)*pow(T,3) + Constant(6.176e-06)*pow(T,2) +Constant(0.0450028)*T + Constant(-242.787))
#Dg_w = ()
#
#entropy of water, hydrogen and oxygen; Consistent with Janaf (tested for 1073K)
# oxygen
s_o = Constant(1.0276975e-09)*4*pow(T,3) + Constant(-9.673333333333333e-06)*3*pow(T,2) + Constant(0.04181925)*2*T + Constant(184.712) + R*ln(p_o*1.0e-5)
# hydrogen
s_h = Constant(1.0906075e-09)*4*pow(T,3) + Constant(-9.820466666666667e-06)*3*pow(T,2) + Constant(0.0400579)*2*T + Constant(111.135) + R*ln(p_h*1.0e-5)
# water
s_w = Constant(1.087365e-09)*4*pow(T,3) + Constant(-1.0228033333333333e-05)*3*pow(T,2) + Constant(0.0468493)*2*T + Constant(165.31) + R*ln(p_w*1.0e-5)

## isobaric heat capacities at 0.1 MPa
#c_p_o    = Constant(31.0)     # oxygen
#c_p_h    = Constant(30.0)     # hydrogen
#c_p_w    = Constant(41.5)     # water vapour
h_w = T*s_w + mu_w_ref + R*T*ln(p_w*1.0e-5)
h_o = T*s_o + mu_o_ref + R*T*ln(p_o*1.0e-5)
h_h = T*s_h + mu_h_ref + R*T*ln(p_h*1.0e-5)
Dh = h_w - h_h - 0.5*h_o

###############################################################################################
##
##    butler-volmer 
##
###############################################################################################
# cathode
p_ref  = Constant(1.0e05) # reference pressure
E_0_c  = mu_o_ref*inv(4.0) # overpotential
i_0_c  = Constant(2.953e07)*exp(Constant(-1.2e05)*inv(R*T))
bv_arg_c = (F*(e_phi - i_phi) - E_0_c)*inv(R*T)
f_c    = char_c*i_0_c*(-pow(p_o, -0.25)*pow(1.e05, 0.25)*exp(1.0*bv_arg_c) + pow(p_o, 0.25)*pow(1.0e-05,0.25)*exp(-1.0*bv_arg_c))    # Butler -- Volmer cathode

# anode
E_0_a  = (mu_w_ref - mu_h_ref)*inv(2.0) # overpotential
bv_arg_a = (F*(e_phi - i_phi) - E_0_a)*inv(R*T)
i_0_a  = Constant(6.949e08)*exp(Constant(-1.0e05)*inv(R*T))
f_a    = char_a*i_0_a*(pow(p_h, 0.5)*pow(p_w, -0.5)*exp(1.0*bv_arg_a)-pow(p_w, 0.5)*pow(p_h, -0.5)*exp(-1.0*bv_arg_a))   # Butler -- Volmer anode



###############################################################################################
##
##    Boundary conditions
##
###############################################################################################
#
#Temperature - boundary temperature setting
T_o = Constant(1.073e03)
t_0 = T_o - float(temp_dif)
T_h = Constant(t_0)
Dg = 0.5*(
-(1.0906075e-09*(pow(T_h,4)-pow(300.0,4)) + (-9.820466666666667e-06)*(pow(T_h,3)-pow(300.0,3)) + 0.0400579*(pow(T_h,2)-pow(300.0,2)) + 111.135*(T_h-300.0)) #hydrogen
-0.5*(1.0276975e-09*(pow(T_o,4)-pow(300.0,4)) + (-9.673333333333333e-06)*(pow(T_o,3)-pow(300.0,3)) + 0.04181925*(pow(T_o,2)-pow(300.0,2)) + 184.712*(T_o-300.0)) #oxygen
-(-228.582e03  -(1.087365e-09*(pow(T_h,4)-pow(300.0,4)) + (-1.0228033333333333e-05)*(pow(T_h,3)-pow(300.0,3)) + (0.0468493)*(pow(T_h,2)-pow(300.0,2)) + (165.31)*(T_h-300.0))) #water
)
print "Dg = ", float(Dg), " J/mol e-\n\n"

##
TT_h = DirichletBC(V.sub(0), T_h, Hydrogen_inlet())
TT_o = DirichletBC(V.sub(0), T_o, Oxygen_inlet())
# electric potential
V_appl = Constant(-float(l_b))  #### applied voltage starting value
##
e_phi_o = DirichletBC(V.sub(2), Constant(0.0), Oxygen_inlet())
e_phi_h = DirichletBC(V.sub(2), V_appl, Hydrogen_inlet())
# pressure
p_o_in = Constant(1.0e05)
p_h_in = Constant(0.5e05)
p_w_out= Constant(0.5e05)
##
p_o_o = DirichletBC(V.sub(3), p_o_in, Oxygen_inlet())
p_h_h = DirichletBC(V.sub(4), p_h_in, Hydrogen_inlet())
p_w_h = DirichletBC(V.sub(5), p_w_out, Hydrogen_inlet())
# list of Dbc conditions
bc = [TT_h, TT_o, e_phi_o, e_phi_h, p_o_o, p_h_h, p_w_h]

print "Boundary conditions defined.\n"

###########################################################################
##
## Variational formulas
##
###########################################################################

## water vapor
#w_flux = -inv(R*T)*(D_k_w*grad(p_w) + p_w*kmu*grad(p_w + p_h)) ## testing flux
#w_flux_obsolete = -( inv(R*T)*inv(inv(D_k_w) + p_h*inv(D_hw*(p_h+p_w)) - D_k_h*p_h*p_w*inv(D_hw*(p_h+p_w)*(D_k_h*p_w + D_hw*(p_h+p_w))))*
#      (p_w*D_k_h*inv(p_w*(D_hw+D_k_h) + D_hw*p_h)*(grad(p_h) + p_h*grad(p_w+p_h)*kmu*inv(D_k_h)) + grad(p_w) + p_w*grad(p_h+p_w)*kmu*inv(D_k_w))
#   )



w_flux = -inv(R*T)*D_k_w*inv(D_k_w*p_h + D_k_h*p_w + (p_w+p_h)*D_hw)*(
               (p_w*D_k_h+ (p_w+p_h)*D_hw)*grad(p_w)
            +   (p_w+p_h)*D_k_h*grad(p_h))
Water =   (  (inner(-w_flux, grad(f)) - inv(F)*L*L*0.5*f_a*active_surf_a*f)*dx_subdomain(1) 
       + Constant(1.0e-12)*inner(grad(p_w), grad(f))*dx_subdomain(2) 
       + Constant(1.0e-12)*inner(grad(p_w), grad(f))*dx_subdomain(3)
    )


## hydrogen
#h_flux = -inv(R*T)*(D_k_h*grad(p_h) + p_h*kmu*grad(p_h + p_w)) ## testing flux
# h_flux_obsolete = -( inv(R*T)*inv(inv(D_k_h)+p_w*inv(D_hw*(p_h+p_w)) - D_k_w*p_h*p_w*inv(D_hw*(p_h + p_w)*(D_k_w*p_h + D_hw*(p_h + p_w))))*
#      (p_h*D_k_w*inv((D_hw + D_k_w)*p_h + D_hw*p_w)*(grad(p_w)+p_w*grad(p_w + p_h)*kmu*inv(D_k_w)) + grad(p_h) + p_h*kmu*inv(D_k_h)*grad(p_w+p_h))
#   )
h_flux = -inv(R*T)*D_k_h*inv(D_k_h*p_w + D_k_w*p_h + (p_w+p_h)*D_hw)*(
                (p_h*D_k_w+ (p_w+p_h)*D_hw)*grad(p_h)
            +   (p_w+p_h)*D_k_w*grad(p_w))
Hydrogen =  (   (inner(-h_flux, grad(e)) + inv(F)*L*L*0.5*f_a*active_surf_a*e)*dx_subdomain(1) 
          + Constant(1.0e-12)*inner(grad(p_h), grad(e))*dx_subdomain(2) 
          + Constant(1.0e-12)*inner(grad(p_h), grad(e))*dx_subdomain(3)
       )



## oxygen
o_flux = -inv(R*T)*(D_k_o + p_o*kmu)*grad(p_o)

Oxygen = (    Constant(1.0e-12)*inner(grad(p_o), grad(d))*dx_subdomain(1) 
       + Constant(1.0e-12)*inner(grad(p_o), grad(d))*dx_subdomain(2)
       + (inner(-o_flux, grad(d))+ inv(F)*L*L*0.25*f_c*active_surf_c*d)*dx_subdomain(3) 

    )


## electrone potential
Electric = ( (sigma_ni*inner(grad(e_phi),grad(c)) + L*L*1.0*f_a*active_surf_a*c )*dx_subdomain(1)#         
       + Constant(1.0e-12)*(inner(grad(e_phi),grad(c)))*dx_subdomain(2)
       +(sigma_lsm*inner(grad(e_phi),grad(c)) - L*L*1.0*f_c*active_surf_c*c)*dx_subdomain(3)# 
)

## ionic potential
Ion = ( 
     (S_i*sigma_i_a*inv(F)*inner(grad(T), grad(b))     + sigma_i_a*inner(grad(i_phi),grad(b)) - L*L*1.0*f_a*active_surf_a*b)*dx_subdomain(1)
    +(S_i*sigma_i_bulk*inv(F)*inner(grad(T), grad(b))  + sigma_i_bulk*inner(grad(i_phi),grad(b)))*dx_subdomain(2)
    +(S_i*sigma_i_c*inv(F)*inner(grad(T), grad(b))    + sigma_i_c*inner(grad(i_phi),grad(b)) + L*L*1.0*f_c*active_surf_c*b)*dx_subdomain(3)
)

## balance of total energy
Energy = (
   
   ( 
   Lambda_a*inner(grad(T),grad(a))
   + T*inv(pow(F,2))*(sigma_i_a*pow(S_i,2))*inner(grad(T), grad(a))
   ##+ T*inv(pow(F,2))*(sigma_ni*pow(S_e,2))*inner(grad(T), grad(a))
   + T*inv(F)*sigma_i_a*S_i*inner(grad(i_phi),grad(a))
   ##+ T*inv(F)*sigma_ni*S_e*inner(grad(e_phi),grad(a))
   + i_phi*sigma_i_a*(S_i*inv(F)*inner(grad(T), grad(a)) + inner(grad(i_phi),grad(a)))
      + i_phi*sigma_i_a*inner(grad(i_phi),grad(a))
   + e_phi*(sigma_ni*inner(grad(e_phi),grad(a)))
   + h_h*inner(-h_flux, grad(a)) 
   + h_w*inner(-w_flux, grad(a))
   )*dx_subdomain(1)
   
   + ( 
   Lambda_e*inner(grad(T),grad(a))
   + T*inv(pow(F,2))*(sigma_i_bulk*pow(S_i,2))*inner(grad(T), grad(a))
   + T*inv(F)*sigma_i_bulk*S_i*inner(grad(i_phi),grad(a))
   + i_phi*(S_i*sigma_i_bulk*inv(F)*inner(grad(T), grad(a)) + sigma_i_bulk*inner(grad(i_phi),grad(a)))
   )*dx_subdomain(2)
   
   +(
   Lambda_c*inner(grad(T),grad(a))
   + T*inv(pow(F,2))*sigma_i_c*pow(S_i,2)*inner(grad(T), grad(a))
   ##+ T*inv(pow(F,2))*sigma_lsm*pow(S_e,2)*inner(grad(T), grad(a))
   + T*inv(F)*sigma_i_c*S_i*inner(grad(i_phi),grad(a))
   ##+ T*inv(F)*sigma_lsm*S_e*inner(grad(e_phi),grad(a))
   #+ i_phi*sigma_i_c*(S_i*inv(F)*inner(grad(T), grad(a)) + inner(grad(i_phi),grad(a)))
   + i_phi*sigma_i_c*inner(grad(i_phi),grad(a))
   + e_phi*(sigma_lsm*inner(grad(e_phi),grad(a)))
   + h_o*inner(-o_flux, grad(a))
   )*dx_subdomain(3)
   
)

# complete variational formula
GG = Energy + Ion + Electric + Oxygen + Hydrogen + Water
print "Variational formula defined\n"

# files for solution storing
file_t       = File("%stemperature.xyz" %(path))
file_i       = File("%si_potential.xyz" %(path))
file_e       = File("%se_potential.xyz" %(path))
file_pw      = File("%swater.xyz" %(path))
file_ph      = File("%shydrogen.xyz" %(path))
file_po      = File("%soxygen.xyz" %(path))
file_bv      = File("%sbv.xyz" %(path))
file_bv_c   = File("%sbv_c.xyz" %(path))
file_bv_a   = File("%sbv_a.xyz" %(path))
file_it      = File("%sit.xyz" %(path)) # flux of ion
file_et      = File("%set.xyz" %(path)) # flux of elec.
file_eta    = File("%seta.xyz" %(path))# overpotential
file_w_flux	    = File("%sw_flux.xyz" %(path))
file_o_flux	    = File("%so_flux.xyz" %(path))
file_h_flux	    = File("%sh_flux.xyz" %(path))
file_h_w = File("%sh_w.xyz" %(path))
file_h_o = File("%sh_o.xyz" %(path))
file_h_h = File("%sh_h.xyz" %(path))
file_Dh = File("%sDh.xyz" %(path))
file_eta    = File("%seta.xyz" %(path))
file_e0c    = File("%se0c.xyz" %(path))
file_e0a    = File("%se0a.xyz" %(path))
file_sigmas_jqp    = File("%ssigmas_jqp.xyz" %(path))
file_sigmas_n    = File("%ssigmas_n.xyz" %(path))
file_sigmas_w    = File("%ssigmas_w.xyz" %(path))
file_sigmas_h    = File("%ssigmas_h.xyz" %(path))
file_sigmas_o    = File("%ssigmas_o.xyz" %(path))
file_sigmas_i    = File("%ssigmas_i.xyz" %(path))
file_sigmas_e    = File("%ssigmas_e.xyz" %(path))
file_sigmas_bv = File("%ssigmas_bv.xyz" %(path))
file_sigmas    = File("%ssigmas.xyz" %(path))
file_jstot= File("%sjstot.xyz" %(path))
file_jqp= File("%sjqp.xyz" %(path))
file_js_jqp= File("%sjs_jqp.xyz" %(path))
file_js_jw= File("%sjs_jw.xyz" %(path))
file_js_jo= File("%sjs_jo.xyz" %(path))
file_js_jh= File("%sjs_jh.xyz" %(path))
file_dT= File("%sdT.xyz" %(path))
file_jen= File("%sjen.xyz" %(path))
file_jen_i= File("%sjen_i.xyz" %(path))
file_jen_e= File("%sjen_e.xyz" %(path))
file_jen_w= File("%sjen_w.xyz" %(path))
file_jen_h= File("%sjen_h.xyz" %(path))
file_jen_o= File("%sjen_o.xyz" %(path))
file_sigmasT    = File("%ssigmasT.xyz" %(path))
file_T0sigmas    = File("%sT0sigmas.xyz" %(path))
file_jstotDT= File("%sjstotDT.xyz" %(path))
file_MOL= File("%sMOL.xyz" %(path))

## file for polarisation curve storage 
pol_c     = open(path+'pol_c'+typ, 'w')
pol_c.close()
tag         = open(path+'taglist', 'w')
tag.close()
#shutil.copy2('./e_s.py', path+"e_s.py") copy of source to /path
#shutil.copy2('./plot.gnu', path+"plot.gnu" ) copy of plot.gnu to /path


###########################################################################
##
## Nonliner variational solver 
##
###########################################################################

#setting starting values for Newton
T, i_phi, e_phi, p_o, p_h, p_w = v.split()

# initializing variables 
T_start      = interpolate(Constant(1.073e03), T_space)
i_phi_start = interpolate(Constant(0.4), I_phi_space)
e_phi_start = interpolate(Constant(0.0), E_phi_space)
p_o_start   = interpolate(Constant(1.0e05), O_space)
p_h_start   = interpolate(Constant(0.5e05), H_space)
p_w_start   = interpolate(Constant(0.5e05), W_space)

assign(T, T_start)
assign(i_phi, i_phi_start)
assign(e_phi, e_phi_start)
assign(p_o, p_o_start)
assign(p_h, p_h_start)
assign(p_w, p_w_start)


print "Starting values set.\n"
print "Solving...\n"


# cycle for V_appl, computing states of FC for different applied voltage, iterated by $step, stops at $stop
#stop = -1.8  # stop value for iteration
#step = .01  # step of V_appl iteration
upper_bound = float(u_b)
lower_bound = float(l_b)
step        = float(krok)
print float(V_appl),'      ',upper_bound,'     ',lower_bound
iteration_counter = 0
while (float(abs(V_appl)) < (upper_bound+DOLFIN_EPS)):
   solve( GG == 0, v, bc, solver_parameters={"newton_solver":{"relative_tolerance": 1.0e-07, "maximum_iterations" :60, "error_on_nonconvergence": False} })
   # boundary condition update and polarization save
   jjj = (-inv(L)*(S_i*inv(F)*T.dx(0) + i_phi.dx(0))*sigma_i_bulk*dx_subdomain(2))
   ## write to polarisation data curve
   pol_c = open(path+'pol_c'+typ, 'a') 
   pol_c.write(str(-float(V_appl))+'   '+str(assemble(jjj))+'\n')
   pol_c.close()
   tag = open(path+'taglist', 'a')
   tag.write(str(float(V_appl))+'       '+str(iteration_counter))
   tag.close()
   iteration_counter = iteration_counter + 1
   ##
      #
   ###########################################################################
   ##
   ##  DATA PROCESSING
   ##
   ###########################################################################
   i_tok = -inv(L)*(S_i*inv(F)*T.dx(0) + i_phi.dx(0))*(sigma_i_a*char_a + sigma_i_c*char_c + sigma_i_bulk*(1.0 - char_a - char_c)) # flux of ions
   e_tok = -inv(L)*(e_phi.dx(0))*(sigma_ni*char_a  + sigma_lsm*char_c + Constant(1.0e-10)*(1.0 - char_a - char_c)) # flux of electrons
   jqp_a = ( 
        Lambda_a*grad(T)
      + T*inv(pow(F,2))*(sigma_i_a*pow(S_i,2))*grad(T)
      #+ T*inv(pow(F,2))*(sigma_ni*pow(S_e,2))*grad(T)
      + T*inv(F)*sigma_i_a*S_i*grad(i_phi)
      #+ T*inv(F)*sigma_ni*S_e*grad(e_phi)
      + i_phi*(S_i*sigma_i_a*inv(F)*grad(T) + sigma_i_a*grad(i_phi))
      + e_phi*sigma_ni*grad(e_phi)
      + h_h*(-h_flux) 
      + h_w*T*(-w_flux)
   )
   jqp_e = (
        Lambda_e*grad(T)
      + T*inv(pow(F,2))*(sigma_i_bulk*pow(S_i,2))*grad(T) 
      + T*inv(F)*sigma_i_bulk*S_i*grad(i_phi) 
      + i_phi*(S_i*sigma_i_bulk*inv(F)*grad(T) + sigma_i_bulk*grad(i_phi))
   )
   jqp_c = (
        Lambda_c*grad(T)
      + T*inv(pow(F,2))*sigma_i_c*pow(S_i,2)*grad(T)
      #+ T*inv(pow(F,2))*sigma_lsm*pow(S_e,2)*inner(grad(T), grad(a))
      + T*inv(F)*sigma_i_c*S_i*grad(i_phi)
      #+ T*inv(F)*sigma_lsm*S_e*inner(grad(e_phi),grad(a))
      + i_phi*(S_i*sigma_i_c*inv(F)*grad(T) + sigma_i_c*grad(i_phi))
      + e_phi*sigma_lsm*grad(e_phi)
      + h_o*(-o_flux)
   )
   itS_a = -S_i*sigma_i_a*inv(F)*grad(T) 
   
   itS_e = -S_i*sigma_i_bulk*inv(F)*grad(T)  
       
   itS_c = -S_i*sigma_i_c*inv(F)*grad(T)

   #plot(D_k_w, interactive=True)
   #plot(D_hw, interactive=True)
   #plot(p_w.dx(0), interactive=True)
   #plot(p_h.dx(0), interactive=True)

   flux_w = -inv(R*T)*D_k_w*inv(D_k_w*p_h + D_k_h*p_w + (p_w+p_h)*D_hw)*(
               (p_w*D_k_h+ (p_w+p_h)*D_hw)*p_w.dx(0)
            +   (p_w+p_h)*D_k_h*p_h.dx(0))
   #plot((D_k_w*p_h + D_k_h*p_w + (p_w+p_h)*D_hw), interactive=True)
   it  = project(i_tok, DG_space)
   et  = project(e_tok, DG_space)
   bv  = project(f_a + f_c, DG_space)
   bv_a   = project(f_a, DG_space)
   bv_c   = project(f_c, DG_space)
   ph  = project(p_h, DG_space)
   pw  = project(p_w, DG_space)
   po  = project(p_o, DG_space)
   ephi   = project(e_phi, DG_space)
   eta_a = char_a*(e_phi - i_phi - inv(F)*E_0_a)
   eta_c = char_c*(e_phi - i_phi - inv(F)*E_0_c)
   eta    = project((eta_c + eta_a), DG_space)
   ec  = project(E_0_c, T_space)
   ea  = project(E_0_a, T_space)
   #measurable heat flux over the whole domain
   jqp = char_a*jqp_a + char_e*jqp_e + char_c*jqp_c
   jqp_x = inner(jqp, Constant([1.0]))
   # entropy production due to heat flux 
   sigmas_jqp =inv(L)*(
       -inv(pow(T,2))*inner(jqp, grad(T))
   )
   #entropy production due to flux of water
   sigmas_w = inv(L)*inv(L)*(-R)*char_a*inv(p_w)*flux_w*p_w.dx(0)

   print '\n Entropy production due to the water transport is ',assemble(L*sigmas_w*dx)
   print 'Test ',assemble(inner(w_flux*char_a, grad(p_w))*dx),'\n'
   #entropy production due to flux of hydrogen
   sigmas_h = inv(L)*inv(L)*(-R*inner(char_a*h_flux, inv(p_h)*grad(p_h)))
   #entropy production due to flux of oxygen  
   sigmas_o = inv(L)*inv(L)*(-R*inner(char_c*o_flux, inv(p_o)*grad(p_o)))
   #entropy production due to diffusion of neutral species
   sigmas_n = sigmas_w + sigmas_h + sigmas_o
   
   #entropy production due to transport of ions
   sigmas_i = inv(L)*(
      -inv(T)*i_tok*i_phi.dx(0)
   )
   #entropy production due to transport of electrons
   sigmas_e = inv(L)*(
      -inv(T)*e_tok*e_phi.dx(0)
   )
   #entropy production due to BuVol at the anode
   sigmas_bv_a = (
      inv(T)*active_surf_a*f_a * (eta_a + 0.5*R*T*inv(F)*ln(p_h*inv(p_w)))
   )
   #entropy production due to BuVol at the anode
   sigmas_bv_c = (
      inv(T)*(-active_surf_c*f_c) * (eta_c + R*T*inv(F)*(-0.25)*ln(p_o*1.0e-5))
   )
   #entropy production due to BuVol
   sigmas_bv = sigmas_bv_a + sigmas_bv_c
   
   #total entropy production
   sigmas = sigmas_jqp + sigmas_n + sigmas_i + sigmas_bv #+ sigmas_e 
   print "Entropy production defined."
   
   sigmas_total = assemble(L*sigmas*dx)
   print "Sigmas_total = ", sigmas_total
   
   
   #entropy flux
   jstot_a = jqp_a*inv(T) + inv(L)*(w_flux*s_w + h_flux*s_h)
   jstot_e = jqp_e*inv(T)
   jstot_c = jqp_c*inv(T) + inv(L)*o_flux*s_o
   jstot = char_a*jstot_a + char_e*jstot_e + char_c*jstot_c
   jstot_total = assemble(inner(jstot,n)*ds(4) + inner(jstot, n)*ds(5))
   js_jqp = inner(jqp*inv(T), Constant([1.0]))
   js_jw = inner(inv(L)*w_flux*s_w*char_a, Constant([1.0]))
   js_jh = inner(inv(L)*h_flux*s_h*char_a, Constant([1.0]))
   js_jo = inner(inv(L)*o_flux*s_o*char_c, Constant([1.0]))
   
   #jstot_vector = project(jstot, VectorFunctionSpace(mesh, 'DG', 1))
   #jstot_x, jstot_y, jstot_y = jstot_vector.split(deepcopy=True)
   jstot_x = inner(jstot, Constant([1.0]))
   
   jen_i = inv(L)*(
           char_a*(-sigma_i_a*grad(i_phi)*i_phi)+
           char_e*(-sigma_i_bulk*grad(i_phi)*i_phi)+
           char_c*(-sigma_i_c*grad(i_phi)*i_phi)
           )
   jen_e = inv(L)*(
           char_a*(-sigma_ni*grad(e_phi)*e_phi)+
           char_c*(-sigma_lsm*grad(e_phi)*e_phi)
           )
   
   jen_h = inv(L)*char_a*(h_h*h_flux)
   jen_w = inv(L)*char_a*(h_w*w_flux)
   jen_o = inv(L)*char_c*(h_o*o_flux)
   jen =jqp + jen_i + jen_e + jen_h + jen_w + jen_o
    
   #map of losses
   MOL = T*sigmas + inv(L)*inner(jstot, grad(T))
   j = assemble(jjj)
   try:
      info_file = open(path+"info.txt", 'w')
      info_file.write("I = "+str(j)+" A/m^2\n")
      info_file.write("V = "+str(float(V_appl))+" V\n") 
      info_file.write("W = "+str(-j*float(V_appl))+" W/m^2\n")
      info_file.write("Jw = "+str(0.5*j/96485) + " mol/m^2s of water\n")
      info_file.write("DG(1073K) = "+str(0.5*j/96485.0*187000.0) + " mol/m^2s of water\n")
      info_file.write("T_h = "+str(float(T_h))+" K\n")
      info_file.write("T_o = "+str(float(T_o))+" K\n")
      info_file.write("sigmas_jqp = "+ str(assemble(L*sigmas_jqp*dx)) + " J/Ksm^2\n")
      info_file.write("sigmas_n = "+ str(assemble(L*sigmas_n*dx)) + " J/Ksm^2\n")
      info_file.write("sigmas_w = "+ str(assemble(L*sigmas_w*dx)) + " J/Ksm^2\n")
      info_file.write("sigmas_h = "+ str(assemble(L*sigmas_h*dx)) + " J/Ksm^2\n")
      info_file.write("sigmas_o = "+ str(assemble(L*sigmas_o*dx)) + " J/Ksm^2\n")
      info_file.write("sigmas_i = "+ str(assemble(L*sigmas_i*dx)) + " J/Ksm^2\n")
      info_file.write("sigmas_e = "+ str(assemble(L*sigmas_e*dx)) + " J/Ksm^2\n")
      info_file.write("sigmas_bv = "+ str(assemble(L*sigmas_bv*dx)) + " J/Ksm^2\n")
      info_file.write("Total entropy production = " + str(sigmas_total) + " J/Ksm^2\n")
      info_file.write("Total entropy flux = " + str(jstot_total) + " J/Ksm^2\n")
      info_file.write("(DeltaG-W)/T0 = "+ str(-(-j*float(V_appl)-float(Dg)*j/96485)/float(T_h))+ " J/Km^2 s\n")
      info_file.write("DeltaG = "+ str(float(Dg)*j/96485)+ " J/m^2 s\n")
      info_file.write("MOL = "+ str(assemble(L*MOL*dx))+ " J/m^2 s\n")
      info_file.write("DeltaG - MOL = "+ str(float(Dg)*j/96485-assemble(L*MOL*dx))+ " J/m^2 s\n")
   finally:
      info_file.close()

   # saving data
   if save:
      file_t     << T
      file_i     << i_phi
      file_e     << ephi
      file_po    << po
      file_pw    << pw
      file_ph    << ph
      file_bv    << bv
      file_it    << it
      file_et    << et
      file_eta   << eta
      file_bv_c     << bv_c
      file_bv_a     << bv_a
   
   
   s_jqp = project(sigmas_jqp, DG_space)
   s_gas = project(sigmas_n, DG_space)
   s_ion = project(sigmas_i, DG_space)
   s_ele = project(sigmas_e, DG_space)
   s_buv = project(sigmas_bv, DG_space)
   s_sig = project(sigmas, DG_space)
   
   file_sigmas_jqp << s_jqp
   file_sigmas_n << s_gas
   file_sigmas_i << s_ion
   file_sigmas_e << s_ele
   file_sigmas_bv << s_buv
   file_sigmas << s_sig
   ##################################################################################### 
   # Pyplot playground
   #####################################################################################
   [pl_i, pl_bv] = [extract_data(mesh, s_ion) + ['-'], extract_data(mesh, s_buv) + ['-']] 
   # coordinates sorting question == my mesh sucks
   #coords = pl_i[0]
   #indices = coords.argsort(0)
   #coords.sort(0)  # mesh coordinates need to be in ascending order
   #pl_i[1] = pl_i[1][indices]  # sort function values accordingly
   #pl_i[0] = coords
   # alternative : beggining trunc
   pl_i[0] = pl_i[0][7:]
   pl_i[1] = pl_i[1][7:]
   pl_bv[0] = pl_bv[0][7:]
   pl_bv[1] = pl_bv[1][7:]
   # maximum cut
   pl_i[1][np.argmax(pl_i[1])] = pl_i[1][np.argmax(pl_i[1])-1]   
   # insertion of NaN into jumps
   pos = np.where(np.abs(np.diff(pl_i[1])) >= 50.0)[0]+1
   pl_i[0] = np.insert(pl_i[0], pos, np.nan)
   pl_i[1] = np.insert(pl_i[1], pos, np.nan)
   pos = np.where(np.abs(np.diff(pl_bv[1])) >= 50.0)[0]+1
   pl_bv[0] = np.insert(pl_bv[0], pos, np.nan)
   pl_bv[1] = np.insert(pl_bv[1], pos, np.nan)
   #
   [pl_q, pl_g, pl_e]  = [extract_data(mesh, s_jqp), extract_data(mesh, s_gas), extract_data(mesh, s_ele)] 
   # trunc
   pl_q[0] = pl_q[0][7:]
   pl_q[1] = pl_q[1][7:]
   pl_g[0] = pl_g[0][7:]
   pl_g[1] = pl_g[1][7:]
   pl_e[0] = pl_e[0][7:]
   pl_e[1] = pl_e[1][7:]
   #plotting 
   plt.rc('text', usetex=True)
   plt.rc('font', family='serif')
   #
   plt.figure(1)
   plt.subplot(211)
   plt.title(r'$\Delta G - T^0\int_V -\frac{1}{T}\left(\nabla\Phi_{\rm i}\cdot{\bf I}_{\rm i} + \dot{\xi}_r\cdot\tilde{\mathcal A}_r + \sum_\alpha(\nabla\mu_\alpha)_T\cdot{\bf j}_\alpha + \nabla\Phi_{\rm e}\cdot{\bf I}_{\rm e}\right)$')
   plt.plot(*pl_i, label = r'ion transport', linewidth = 4, linestyle = '-', marker="o", markevery = 80, ms = 6) 
   plt.plot(*pl_bv, label = r'chemical reaction', linewidth = 4, linestyle = '-')
   plt.axvline(x=0.0, ymin=0.0, ymax=400, linewidth=0.5, color = 'black', linestyle = '--' )
   plt.axvline(x=1.0, ymin=0.0, ymax=400, linewidth=0.5, color = 'black', linestyle = '--')
   plt.legend(loc=2, borderaxespad=3.0, frameon = False)
   plt.xticks(list(plt.xticks()[0]) + [-14.0, 1.0])
   plt.xlim(-14.0,5.0)
   plt.ylabel(r'${\rm W}/{\rm m}^3$')
   #
   plt.subplot(212)
   plt.plot(*pl_q, label =r'heat transfer', lw = 4,marker="o", markevery = 80, ms = 6)
   plt.plot(*pl_g, label = r'gas transport', lw = 4)
   plt.plot(*pl_e, label = r'electric current', lw = 4)
   plt.xlabel(r'$x/\, 10^{-4}{\rm m}$')
   plt.ylabel(r'${\rm W}/{\rm m}^3$')
   plt.axvline(x=0.0, ymin=0.0, ymax=400, linewidth=0.5, color = 'black', linestyle = '--' )
   plt.axvline(x=1.0, ymin=0.0, ymax=400, linewidth=0.5, color = 'black', linestyle = '--')
   plt.xticks(list(plt.xticks()[0]) + [-14.0, 1.0])
   plt.xlim(-14.0,5.0)
   plt.ylim(-0.01, .180)
   plt.legend(loc=2, borderaxespad=3.0, frameon=False)
   plt.show()
   plt.savefig("%ssigmas.eps" %(path))
   #####################################################################################
   #####################################################################################
   file_sigmas_w << project(sigmas_w, DG_space)
   file_sigmas_h << project(sigmas_h, DG_space)
   file_sigmas_o << project(sigmas_o, DG_space)
 
   file_jqp<< project(jqp_x, DG_space)
   file_jstot << project(jstot_x, DG_space)
   file_o_flux << project(inner(inv(L)*o_flux, Constant([1.0])), DG_space)
   file_h_flux << project(inner(inv(L)*h_flux, Constant([1.0])), DG_space)
   file_w_flux << project(inner(inv(L)*w_flux, Constant([1.0])), DG_space)
   file_h_w << project(h_w, DG_space)
   file_h_o << project(h_o, DG_space)
   file_h_h << project(h_h, DG_space)
   file_Dh << project(Dh, DG_space)
   file_js_jqp << project(js_jqp, DG_space)
   file_js_jw<< project(js_jw, DG_space)
   file_js_jh<< project(js_jh, DG_space)
   file_js_jo<< project(js_jo, DG_space)
   file_dT<< project(inner(grad(T), Constant([1.0])), DG_space)
   file_jen<< project(inner(jen, Constant([1.0])), DG_space)
   file_jen_w<< project(inner(jen_w, Constant([1.0])), DG_space)
   file_jen_h<< project(inner(jen_h, Constant([1.0])), DG_space)
   file_jen_o<< project(inner(jen_o, Constant([1.0])), DG_space)
   file_jen_i<< project(inner(jen_i, Constant([1.0])), DG_space)
   file_jen_e<< project(inner(jen_e, Constant([1.0])), DG_space)
   file_sigmasT << project(T*sigmas, DG_space)
   file_T0sigmas<< project(T_h*sigmas, DG_space)
   file_jstotDT<< project(inner(jstot, inv(L)*grad(T)), DG_space)
   file_MOL << project(MOL, DG_space)

   # cleaning for new iteration
   T_start     = interpolate(Constant(1.073e03), T_space)
   i_phi_start = interpolate(Constant(0.4), I_phi_space)
   e_phi_start = interpolate(Constant(0.0), E_phi_space)
   p_o_start   = interpolate(Constant(1.0e05), O_space)
   p_h_start   = interpolate(Constant(0.5e05), H_space)
   p_w_start   = interpolate(Constant(0.5e05), W_space)
   
   assign(T, T_start)
   assign(i_phi, i_phi_start)
   assign(e_phi, e_phi_start)
   assign(p_o, p_o_start)
   assign(p_h, p_h_start)
   assign(p_w, p_w_start)
   
   V_appl = Constant(float(V_appl) - step)    # V_appl iteration
   #print V_appl,' ',float(V_appl),''
   kond = DirichletBC(V.sub(2), V_appl, Hydrogen_inlet()) # bc e_phi update
   bc = [TT_h, TT_o, e_phi_o, kond, p_o_o, p_h_h, p_w_h]



