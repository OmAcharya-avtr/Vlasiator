import numpy as np
from scipy.interpolate import RegularGridInterpolator
from numba import jit, prange
import multiprocessing as mp
from functools import partial

class SemiLagrangianAdvection:
    def __init__(self, grid, n_threads=None):
        """
        Initialize the semi-Lagrangian advection solver.
        
        Parameters:
        -----------
        grid : Grid
            Grid object containing spatial and velocity information
        n_threads : int, optional
            Number of threads to use for parallel processing. If None, uses all available cores.
        """
        self.grid = grid
        self.n_threads = n_threads if n_threads is not None else mp.cpu_count()
        # Pre-allocate arrays for better memory management
        self.x_char = np.zeros((grid.nx, grid.nv))
        self.v_char = np.zeros((grid.nx, grid.nv))
        self.a_field = np.zeros((grid.nx, grid.nv))
        # Create interpolator once and reuse
        self._setup_interpolator()
        
    def _setup_interpolator(self):
        """Setup the interpolator with optimal settings."""
        self.interpolator = RegularGridInterpolator(
            (self.grid.x, self.grid.v),
            np.zeros((self.grid.nx, self.grid.nv)),  # Dummy array, will be updated in advect
            method='linear',
            bounds_error=False,
            fill_value=0.0
        )
    
    @staticmethod
    @jit(nopython=True, parallel=True)
    def _compute_characteristics_numba(x, v, dt, a, x_char, v_char, a_field):
        """Numba-accelerated characteristic computation."""
        nx, nv = x_char.shape
        for i in prange(nx):
            for j in range(nv):
                a_field[i, j] = a[i]
                x_char[i, j] = x[i] - v[j] * dt
                v_char[i, j] = v[j] - a_field[i, j] * dt
        return x_char, v_char
    
    @staticmethod
    @jit(nopython=True)
    def _apply_boundary_conditions(x_char, v_char, x_min, x_max, v_min, v_max, x_range):
        """Apply boundary conditions using Numba."""
        # Periodic in x
        x_char = np.mod(x_char - x_min, x_range) + x_min
        # Reflecting in v
        v_char = np.clip(v_char, v_min, v_max)
        return x_char, v_char
    
    def _interpolate_chunk(self, chunk_indices, f, x_char, v_char):
        """Interpolate a chunk of the distribution function."""
        i_start, i_end = chunk_indices
        f_chunk = np.zeros((i_end - i_start, self.grid.nv))
        for i in range(i_start, i_end):
            for j in range(self.grid.nv):
                f_chunk[i-i_start, j] = self.interpolator((x_char[i, j], v_char[i, j]))
        return f_chunk
    
    def advect(self, f, dt, a):
        """
        Advect the distribution function using optimized semi-Lagrangian method.
        
        Parameters:
        -----------
        f : ndarray
            Current distribution function f(x,v)
        dt : float
            Time step
        a : ndarray
            Acceleration at each spatial point
            
        Returns:
        --------
        f_new : ndarray
            Updated distribution function
        """
        # Update interpolator with current distribution
        self.interpolator.values = f
        
        # Compute characteristics using Numba-accelerated function
        self.x_char, self.v_char = self._compute_characteristics_numba(
            self.grid.x, self.grid.v, dt, a,
            self.x_char, self.v_char, self.a_field
        )
        
        # Apply boundary conditions
        self.x_char, self.v_char = self._apply_boundary_conditions(
            self.x_char, self.v_char,
            self.grid.x[0], self.grid.x[-1],
            self.grid.v[0], self.grid.v[-1],
            self.grid.x[-1] - self.grid.x[0]
        )
        
        # Parallel interpolation
        chunk_size = self.grid.nx // self.n_threads
        chunk_indices = [(i, min(i + chunk_size, self.grid.nx)) 
                        for i in range(0, self.grid.nx, chunk_size)]
        
        with mp.Pool(self.n_threads) as pool:
            interpolate_func = partial(self._interpolate_chunk, f=f, 
                                     x_char=self.x_char, v_char=self.v_char)
            f_chunks = pool.map(interpolate_func, chunk_indices)
        
        # Combine chunks
        f_new = np.vstack(f_chunks)
        
        return f_new
    
    def compute_cfl_dt(self, a):
        """
        Compute the maximum stable time step based on CFL condition.
        Optimized version using vectorized operations.
        """
        # Vectorized computation of maximum velocity and acceleration
        max_vel = np.abs(self.grid.v).max()
        max_acc = np.abs(a).max()
        
        # Compute CFL conditions using vectorized operations
        dx_dt = max(max_vel, 1.0) / self.grid.dx
        dv_dt = max(max_acc, 1.0) / self.grid.dv
        
        # Use the more restrictive condition with a safety factor
        dt_max = 0.5 / max(dx_dt, dv_dt)
        
        # Ensure dt is within reasonable bounds
        return np.clip(dt_max, 1e-6, 0.1) 