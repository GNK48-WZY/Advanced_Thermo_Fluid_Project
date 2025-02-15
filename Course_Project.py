import numpy as np
import dedalus.public as d3
import logging
logger = logging.getLogger(__name__)

# Parameters (Elasto-inertial case S1)
Re = 1000          # Reynolds number
Wi = 30            # Weissenberg number
beta = 0.7         # Solvent viscosity ratio
Lmax = 200         # Maximum polymer extensibility
epsilon = 4e-5     # Polymer stress diffusivity
Ly = 2             # Channel height
Lx = 2*np.pi/50    # Domain length
Nx, Ny = 12, 128   # Resolution
dealias = 3/2      # Dealias factor
stop_sim_time = 50 # Simulation stop time
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'y')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
ybasis = d3.Chebyshev(coords['y'], size=Ny, bounds=(-Ly/2, Ly/2), dealias=dealias)

# Fields
u = dist.VectorField(coords, name='u', bases=(xbasis, ybasis))
p = dist.Field(name='p', bases=(xbasis, ybasis))
Cxx = dist.Field(name='Cxx', bases=(xbasis, ybasis))
Cxy = dist.Field(name='Cxy', bases=(xbasis, ybasis))
Cyy = dist.Field(name='Cyy', bases=(xbasis, ybasis))

# Substitutions
x, y = dist.local_grids(xbasis, ybasis)
ex, ey = coords.unit_vector_fields(dist)

# FENE-P Model Terms
trC = Cxx + Cyy
f = 1 / ((1 - (trC - 3))/Lmax**2)  # Avoid division by zero
Txx = (Cxx * f - 1) / Wi
Txy = (Cxy * f) / Wi
Tyy = (Cyy * f - 1) / Wi
div_T = (1 - beta) * (dx(Txx) + dy(Txy)) * ex + (dx(Txy) + dy(Tyy)) * ey

# Problem Setup
problem = d3.IVP([u, p, Cxx, Cxy, Cyy], namespace=locals())

# Momentum equation (linear LHS)
problem.add_equation("Re*dt(u) + Re*grad(p) - beta*div(grad(u)) = Re*dot(u, grad(u)) + div_T")


# Incompressibility condition
problem.add_equation("div(u) = 0")

# Polymer evolution equations (linear LHS)
problem.add_equation(
    "dt(Cxx) - epsilon*(dx(dx(Cxx)) + dy(dy(Cxx))) = "
    "2*(Cxx*dx(u@x) + Cxy*dy(u@x)) - dot(u, grad(Cxx)) - Txx"
)

problem.add_equation(
    "dt(Cxy) - epsilon*(dx(dx(Cxy)) + dy(dy(Cxy))) = "
    "(Cxx*dy(u@x) + Cyy*dx(u@y)) - dot(u, grad(Cxy)) - Txy"
)

problem.add_equation(
    "dt(Cyy) - epsilon*(dx(dx(Cyy)) + dy(dy(Cyy))) = "
    "2*(Cxy*dx(u@y) + Cyy*dy(u@y)) - dot(u, grad(Cyy)) - Tyy"
)

# Boundary Conditions
problem.add_bc("left(u@x) = 0")
problem.add_bc("right(u@x) = 0")
problem.add_bc("left(u@y) = 0")
problem.add_bc("right(u@y) = 0")
problem.add_bc("left(Cxx) = 1")
problem.add_bc("right(Cxx) = 1")
problem.add_bc("left(Cxy) = 0")
problem.add_bc("right(Cxy) = 0")
problem.add_bc("left(Cyy) = 1")
problem.add_bc("right(Cyy) = 1")

# Solver Setup
solver = problem.build_solver(d3.RK222)
solver.stop_sim_time = stop_sim_time

# Initial Conditions
u['g'][0] = 1 - (y/(Ly/2))**2 + np.random.randn(*u['g'][0].shape) * 1e-6
u['g'][1] = np.random.randn(*u['g'][1].shape) * 1e-6
Cxx['g'] = 1 + np.random.randn(*Cxx['g'].shape) * 1e-6
Cyy['g'] = 1 + np.random.randn(*Cyy['g'].shape) * 1e-6
Cxy['g'] = np.random.randn(*Cxy['g'].shape) * 1e-6

# Main Loop
dt = 1e-4
try:
    while solver.proceed:
        solver.step(dt)
        if solver.iteration % 100 == 0:
            logger.info(f"Iteration: {solver.iteration}, Time: {solver.sim_time:.3f}, dt: {dt:.2e}")
except:
    logger.error("Exception raised, exiting.")
    raise
