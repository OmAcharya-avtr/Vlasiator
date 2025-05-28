import numpy as np

class Grid:
    def __init__(self, nx, nv, x_min, x_max, v_min, v_max):
        """
        Initialize the grid for the Vlasov simulation.
        All quantities are in normalized units:
        - x_min, x_max: in Debye lengths (λ_D)
        - v_min, v_max: in thermal velocities (v_th)
        """
        # Spatial grid
        self.nx = nx
        self.x_min = x_min
        self.x_max = x_max
        self.dx = (x_max - x_min) / (nx - 1)
        self.x = np.linspace(x_min, x_max, nx)
        
        # Velocity grid
        self.nv = nv
        self.v_min = v_min
        self.v_max = v_max
        self.dv = (v_max - v_min) / (nv - 1)
        self.v = np.linspace(v_min, v_max, nv)
        
        # Create 2D meshgrid for calculations
        self.X, self.V = np.meshgrid(self.x, self.v, indexing='ij')
    
    def initialize_maxwellian(self, n0=1.0, v0=0.0, vt=1.0, x0=None, sigma=1.0):
        """
        Initialize a Maxwellian distribution function.
        All parameters are in normalized units:
        - n0: background density (in n_0)
        - v0: drift velocity (in v_th)
        - vt: thermal velocity (normalized to 1)
        - x0: center of perturbation (in λ_D)
        - sigma: width of perturbation (in λ_D)
        """
        if x0 is None:
            x0 = (self.x_max + self.x_min) / 2
        
        # Create spatial perturbation
        x_pert = np.exp(-(self.x - x0)**2 / (2 * sigma**2))
        
        # Create Maxwellian in velocity space
        v_dist = np.exp(-(self.V - v0)**2 / (2 * vt**2)) / np.sqrt(2 * np.pi * vt**2)
        
        # Combine to get full distribution
        f = n0 * x_pert[:, np.newaxis] * v_dist
        
        return f
    
    def get_density(self, f):
        """Compute density by integrating over velocity space."""
        return np.trapz(f, self.v, axis=1)
    
    def get_velocity(self, f):
        """Compute mean velocity."""
        density = self.get_density(f)
        # Avoid division by zero
        density[density < 1e-10] = 1e-10
        return np.trapz(self.V * f, self.v, axis=1) / density
    
    def get_velocity_squared(self, f):
        """Compute mean squared velocity."""
        density = self.get_density(f)
        # Avoid division by zero
        density[density < 1e-10] = 1e-10
        return np.trapz(self.V**2 * f, self.v, axis=1) / density
    
    def get_total_particles(self, f):
        """Compute total number of particles."""
        return np.trapz(np.trapz(f, self.v, axis=1), self.x)
    
    def get_energy(self, f):
        """Compute total energy (kinetic + potential)."""
        # Kinetic energy
        v2 = self.get_velocity_squared(f)
        density = self.get_density(f)
        kinetic = 0.5 * np.trapz(density * v2, self.x)
        
        return kinetic  # In normalized units 