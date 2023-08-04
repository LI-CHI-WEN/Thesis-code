#########################################################################
# ==============================================
# Structure of this project
# ==============================================
# 1. Run name & detail setting
# 2. Run type, output frequency, domain size & (coef / forcing coef) setting
# 3. Giving Background field (forcing)
# 4. Create problem and define variables to do iteration
# 5. Equation set
# 6. Build solver & integration setting (give time scheme) & CFL
# 7. Give initial condition
# 8. Create file directory & start to output h5 file / Plot initial condition & save code
# 9. Main integration loop
#########################################################################
# Topics in Independent Research
# import packages
import numpy as np
import matplotlib.pyplot as plt
import time
from dedalus import public as de
from dedalus.extras import flow_tools
from dedalus.core import operators
import logging
import h5py
import os
import shutil

#########################################################################
# ==============================================
# Detial will be writed
# ==============================================
detail = ['Beta plane turbulence']
         
#########################################################################

def All_coef():
    # ==============================================
    # Run setting
    # ==============================================    
    global R_name
    R_name = "beta02_kappa01"
    
    # ==============================================
    # Output setting
    # ==============================================
    global interval, simulationTime, outputzetas
    interval       = 0.04       # sim_dt [s] Output interval
    simulationTime = 10         # [s]
    outputzetas = False
    
    # ==============================================
    # Set Domain Size
    # ==============================================
    global L, Lx, Ly, nx, ny 
    scale = 2*np.pi
    L = 1
    nx, ny = (512, 512)          # Follow SNC
    Lx, Ly = (scale*L, scale*L)
    # ==============================================
    # Set Coef
    # ==============================================
    global Ld_L, nu_tilt, drag_tilt, beta_tilt
    nu_tilt = 1e-17              # Follow SNC
    Ld_L = 0.02
    F_drag = 0.1    # given in GF21
    F_beta = 0.2    # given in GF21
    
    drag_tilt = F_drag/Ld_L
    beta_tilt = F_beta*(1/Ld_L)**2
    # ==============================================
    # Set forcing Coef
    # ==============================================
    # For spectral space random forcing 
    global Am, rand, kf, Kf, dKf
    Am = 1e-13                             # CAUTION! This determines the energy injection rate
#    kf = 100                               # 32 wave in domain 
#    Kf = 2.*np.pi*kf/Lx                    # forcing at wavenumber 32
#    dKf = 2.*np.pi*30./Lx                  # range of Kf

All_coef()

#########################################################################
# ==============================================
# Create bases and domain
# ==============================================

def domain_info():
    global x_basis, y_basis, domain, x, y, xx, yy
    x_basis = de.Fourier('x', nx, interval=(-Lx/2, Lx/2), dealias=3/2)
    y_basis = de.Fourier('y', ny, interval=(-Ly/2, Ly/2), dealias=3/2)
    domain  = de.Domain([x_basis, y_basis], grid_dtype = np.float64)
    
    x = domain.grid(0)
    y = domain.grid(1)
    yy,xx = np.meshgrid(np.array(y[0,:]),np.array(x[:,0]))

domain_info()


#########################################################################
# ==============================================
# Background field
# Forcing Function
# ==============================================
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand   = np.random.RandomState(seed=23)
noise  = rand.standard_normal(gshape)[slices]

tmpns = Am*( noise - np.mean(noise) )                      

#########################################################################
# ==============================================
# Create Problem and define variables to do iteration
# ==============================================
problem = de.IVP(domain, variables=['psi1','psi2'])

#########################################################################
# ==============================================
# Set  parameters & Substitutions
# ==============================================

problem.parameters['L_Ld']  = 1/Ld_L
problem.parameters['nu_tilt'] = nu_tilt    # viscous term
problem.parameters['drag_tilt']  = drag_tilt
problem.parameters['beta_tilt']  = beta_tilt

# Non-dim operater
problem.substitutions['J(a,b)'] = "dx(a)*dy(b) - dy(a)*dx(b) "
problem.substitutions['Lap(a)']  = "d(a,x=2)+d(a,y=2)"
problem.substitutions['Lap8(a)']  = "Lap(Lap(Lap(Lap(a))))"

problem.substitutions['u1'] = " -dy(psi1) "  # non_dim u
problem.substitutions['u2'] = " -dy(psi2) "
problem.substitutions['v1'] = "  dx(psi1) "  # non_dim v
problem.substitutions['v2'] = "  dx(psi2) "
  
problem.substitutions['zeta1'] = "Lap(psi1) "    # non_dim zeta
problem.substitutions['zeta2'] = "Lap(psi2) "
problem.substitutions['q1'] = "zeta1 + (psi2 - psi1)/2*L_Ld*L_Ld"  # non_dim q
problem.substitutions['q2'] = "zeta2 - (psi2 - psi1)/2*L_Ld*L_Ld"
problem.substitutions['Q1y'] = "beta_tilt + L_Ld*L_Ld"          # non_dim U1^hat-U2^hat = U1/U-U2/U = 2
problem.substitutions['Q2y'] = "beta_tilt -L_Ld*L_Ld"
# ==============================================
# Equation sets
# 2 layer SW Phillips model on Fourier basis
# ==============================================

# u_real = u+U1, so the advection term is not just J(psi,q)
#problem.add_equation("dt(q1) + HD(zeta1) + U1*dx(q1) + v1*Q1y = -J(psi1,q1) ", condition="(nx != 0) or  (ny != 0)")
#problem.add_equation("dt(q2) + HD(zeta2) + gamma*zeta2 + U2*dx(q2) + v2*Q2y = -J(psi2,q2) ", condition="(nx != 0) or  (ny != 0)")
#problem.add_equation("psi1 = 0",                                         condition="(nx == 0) and (ny == 0)")
#problem.add_equation("psi2 = 0",                                         condition="(nx == 0) and (ny == 0)")
problem.add_equation("dt(q1) + nu_tilt*Lap8(q1) + dx(q1) + v1*Q1y = -J(psi1,q1) ", condition="(nx != 0) or  (ny != 0)")
# drag multiplied by a factor of 2, to be consistent with GF20
problem.add_equation("dt(q2) + nu_tilt*Lap8(q2) + 2*drag_tilt*zeta2 - dx(q2) + v2*Q2y = -J(psi2,q2) ", condition="(nx != 0) or  (ny != 0)")
problem.add_equation("psi1 = 0",                                         condition="(nx == 0) and (ny == 0)")
problem.add_equation("psi2 = 0",                                         condition="(nx == 0) and (ny == 0)")
     
#########################################################################
# ==============================================
# Integration setting
# Timestepping
# ==============================================
ts = de.timesteppers.RK443

# Build solver
solver = problem.build_solver(ts)
solver.stop_sim_time = simulationTime
solver.stop_wall_time = np.inf
solver.stop_iteration = np.inf 

# ==============================================
# Now we set integration parameters and the CFL.
# ==============================================
dt = 0.1*Lx/nx*0.01             # constant dt
#CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=10, safety=2,
#                   max_change=1.2, min_change=0.5, max_dt=3*dt)
#CFL.add_velocities(('u1', 'v1'))

flow = flow_tools.GlobalFlowProperty(solver, cadence=50)
flow.add_property("0.5*Lap(psi1)**2", name='KE')

#########################################################################
# ==============================================
# Initial Condition
# ==============================================
psi1  = solver.state['psi1']
psi1['g']  = tmpns
psi2  = solver.state['psi2']
psi2['g']  = tmpns
"""
# SNC
gshape = domain.dist.grid_layout.global_shape(scales=1)
slices = domain.dist.grid_layout.slices(scales=1)
rand   = np.random.RandomState(seed=23)
noise  = rand.standard_normal(gshape)[slices]
#psi1['g']  = noise*U1*2./np.sqrt(f0*f0/gprm/H1)*1.e-3
#psi2['g']  = noise*U1*2./np.sqrt(f0*f0/gprm/H1)*1.e-3
tmpns = 1.e-13*( noise - np.mean(noise) )
psi1['g']  = tmpns   # use different seed?
psi2['g']  = tmpns
"""
#########################################################################
# ==============================================
# Output variables and create file directory
# ==============================================
snap = solver.evaluator.add_file_handler(R_name, sim_dt = interval, max_writes = 600)
snap.add_system(solver.state, layout='g')
snap.add_task("zeta1", layout='g', name="zeta1")
snap.add_task("zeta2", layout='g', name="zeta2")
snap.add_task("u1", layout='g', name="u1")
snap.add_task("u2", layout='g', name="u2")
snap.add_task("v1", layout='g', name="v1")
snap.add_task("v2", layout='g', name="v2")  
# ==============================================
# Plot initial vars
# ==============================================
# Init_plot(psi1=psi1['g'], psi2 = psi2['g'],savefig = True) 

# ==============================================
# Copy code
# ==============================================
C = open(R_name+'/code.py','a')
shutil.copy2(os.path.basename(__file__),R_name+'/code.py')
C.close()

# ==============================================
# Save information into txt
# ==============================================
if 0:
  txt = open(R_name+'/parameters.txt','a')
  txt.write('note\n')
  for De in detail:
      txt.write(De+'\n')
  txt.write('==============================\n\n')
  txt.write('simulation time                   {:3d} s\n'.format(int(simulationTime)))
  txt.write('output interval                   {:3d} hours\n\n'.format(int(interval/3600)))
  txt.write('domain size (Lx)                  {:3.2e}\n'.format(Lx))
  txt.write('Ld^-1 (kd)                        {:3.2e}\n'.format(kd)) 
  txt.write('grid (nx)                         {:3.2e}\n'.format(nx))
  txt.write('hyperdiffusion coefficient (D)    {:3.2e}\n'.format(D)) 
  txt.write('Ekman darg (gamma)                {:3.2e}\n'.format(gamma))
  txt.write('f0                                {:3.2e}\n'.format(f0))
  txt.write('Layer 1 mean flow (U1)            {:3.2e}\n'.format(U1))
#    txt.write('latitude                   (phi)    {:3.2e}\n'.format(phi))
#    txt.write('Layer 1 thickness          (H1)    {:3.2e}\n'.format(H1))
#    txt.write('Layer 2 thickness          (H2)    {:3.2e}\n'.format(H2))
  txt.close()

#########################################################################
# ==============================================
# Main Integration loop
# ==============================================
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
logger.info('Starting loop')
start_run_time = time.time()
#forcing_func.original_args = [dt]

plotted = True
NOTnan = True

i = 0
while (solver.ok) and NOTnan:
    #dt = CFL.compute_dt()
    #forcing_func.args = [dt]
    solver.step(dt)
    
    if solver.iteration % 50 == 0:
        logger.info('Iteration: %i, Time: %4.3f s, dt: %3.1e sec' %(solver.iteration, solver.sim_time, dt))
        logger.info('Total KE = %.e' %flow.max('KE'))
        if np.isnan(flow.max('KE')):
            NOTnan = False
    """
    if solver.sim_time >= i and solver.sim_time < i+interval/2 :
        print('plot zeta '+str(i))
        # Draw day-i zeta
        psi1 = solver.state['psi1']
        zeta1_tmp = de.operators.differentiate(psi1,x=2) + de.operators.differentiate(psi1,y=2)
        zeta1   = zeta1_tmp.evaluate()
        zeta1.set_scales(1,keep_data=True)
        
        Z = zeta1['g']
        Z99 = np.percentile(Z.flatten(),[99])
        
        fig,ax = plt.subplots()
        ax.set_aspect('equal')
        P = ax.pcolormesh(xx,yy,Z,vmax=Z99,vmin=-Z99,cmap='RdBu',shading='auto')

        ax.set_title('Zeta'+',  time ~ {:4.2f} s'.format(i),fontsize=14)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(P,orientation = "horizontal")
        plt.savefig(R_name+'/lastzeta.png')
        plt.close()
        i = i+ interval
    """
          


