import numpy as np
from scipy.fft import fft, ifft

class ElectricField:
    def __init__(self, grid, field_type='sinusoidal', **kwargs):
        """
        Initialize the electric field calculator.
        
        Parameters:
        -----------
        grid : Grid
            Grid object containing spatial and velocity information
        field_type : str
            Type of electric field ('sinusoidal', 'fixed', or 'self_consistent')
        **kwargs : dict
            Additional parameters for specific field types
        """
        self.grid = grid
        self.field_type = field_type
        self.params = kwargs
        
        # Physical constants (in normalized units)
        # Using normalized units where:
        # - Length is normalized by Debye length
        # - Time is normalized by plasma frequency
        # - Velocity is normalized by thermal velocity
        # - Electric field is normalized by k_B*T/(e*lambda_D)
        self.epsilon0 = 1.0  # Normalized permittivity
        self.q = 1.0        # Normalized charge
        self.m = 1.0        # Normalized mass
        
    def compute_field(self, f=None):
        """
        Compute the electric field based on the chosen type.
        
        Parameters:
        -----------
        f : ndarray, optional
            Distribution function for self-consistent field calculation
            
        Returns:
        --------
        E : ndarray
            Electric field at each spatial point (in normalized units)
        """
        if self.field_type == 'sinusoidal':
            return self._sinusoidal_field()
        elif self.field_type == 'fixed':
            return self._fixed_field()
        elif self.field_type == 'self_consistent':
            if f is None:
                raise ValueError("Distribution function required for self-consistent field")
            return self._self_consistent_field(f)
        else:
            raise ValueError(f"Unknown field type: {self.field_type}")
    
    def _sinusoidal_field(self):
        """Compute a sinusoidal electric field in normalized units."""
        k = self.params.get('k', 2*np.pi/self.grid.x[-1])
        E0 = self.params.get('E0', 0.1)  # Reduced default amplitude
        return E0 * np.sin(k * self.grid.x)
    
    def _fixed_field(self):
        """Compute a constant electric field in normalized units."""
        E0 = self.params.get('E0', 0.1)  # Reduced default amplitude
        return np.ones_like(self.grid.x) * E0
    
    def _self_consistent_field(self, f):
        """
        Compute self-consistent electric field using Poisson's equation.
        Uses spectral method for periodic boundary conditions.
        All quantities are in normalized units.
        """
        # Calculate charge density
        rho = self.q * (self.grid.get_density(f) - self.params.get('n0', 1.0))
        
        # Solve Poisson's equation in Fourier space
        k = 2 * np.pi * np.fft.fftfreq(len(self.grid.x), self.grid.dx)
        k[0] = 1e-10  # Avoid division by zero
        
        # Compute potential in Fourier space
        phi_k = fft(rho) / (self.epsilon0 * k**2)
        phi_k[0] = 0  # Set average potential to zero
        
        # Transform back to real space
        phi = np.real(ifft(phi_k))
        
        # Compute electric field
        E = -np.gradient(phi, self.grid.dx)
        
        return E
    
    def get_acceleration(self, E):
        """
        Compute acceleration a = qE/m for particles.
        All quantities are in normalized units.
        
        Parameters:
        -----------
        E : ndarray
            Electric field at each spatial point
            
        Returns:
        --------
        a : ndarray
            Acceleration at each spatial point
        """
        # In normalized units, q/m = 1
        return E 