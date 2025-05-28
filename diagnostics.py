import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import os
import pyspedas
import pytplot

class Diagnostics:
    def __init__(self, grid):
        """Initialize diagnostics with normalized units."""
        self.grid = grid
        self.setup_plots()
        
    def setup_plots(self):
        """Set up the plotting environment."""
        # plt.style.use('seaborn')  # Removed to avoid error
        self.fig_phase = plt.figure(figsize=(20, 16))
        self.ax_phase = self.fig_phase.add_subplot(111)
        self.ax_phase.set_xlabel('x (λ_D)')
        self.ax_phase.set_ylabel('v (v_th)')
        self.ax_phase.set_title('Phase Space Distribution')
        
        self.fig_moments = plt.figure(figsize=(24, 8))
        self.ax_density = self.fig_moments.add_subplot(131)
        self.ax_velocity = self.fig_moments.add_subplot(132)
        self.ax_temperature = self.fig_moments.add_subplot(133)
        
        self.ax_density.set_xlabel('x (λ_D)')
        self.ax_density.set_ylabel('n (n_0)')
        self.ax_density.set_title('Density')
        
        self.ax_velocity.set_xlabel('x (λ_D)')
        self.ax_velocity.set_ylabel('v (v_th)')
        self.ax_velocity.set_title('Velocity')
        
        self.ax_temperature.set_xlabel('x (λ_D)')
        self.ax_temperature.set_ylabel('T (T_0)')
        self.ax_temperature.set_title('Temperature')
        
        plt.tight_layout()
    
    def load_mms_magnetic_field(self, trange, probe='1', data_level='l2', instrument='fgm'):
        """
        Load MMS magnetic field data using pyspedas.
        
        Parameters:
        -----------
        trange : list of str
            Time range for data loading, e.g., ['2015-10-30/06:00:00', '2015-10-30/07:00:00']
        probe : str
            MMS probe number ('1', '2', '3', or '4')
        data_level : str
            MMS data level ('l1', 'l2', 'l3', etc.)
        instrument : str
            MMS instrument ('fgm', 'scm', etc.)
            
        Returns:
        --------
        tplot_vars : list of str
            List of loaded tplot variables.
        """
        print(f"\nLoading MMS data for probe {probe} ({instrument}, {data_level})...")
        tplot_vars = pyspedas.mms.fgm(
            trange=trange,
            probe=probe
        )
        print(f"Loaded tplot variables: {tplot_vars}")
        return tplot_vars

    def plot_time_series_comparison(self, sim_time, sim_data, mms_tplot_vars, sim_label='Simulation Data'):
        """
        Plot simulation time series data alongside MMS data.
        
        Parameters:
        -----------
        sim_time : ndarray
            Simulation time array.
        sim_data : ndarray
            Simulation data time series.
        mms_tplot_vars : list of str
            List of tplot variables containing MMS data.
        sim_label : str
            Label for the simulation data.
        """
        print("\nPlotting time series comparison...")
        
        fig, axes = plt.subplots(nrows=len(mms_tplot_vars) + 1, figsize=(12, 8), sharex=True)
        
        # Plot simulation data
        axes[0].plot(sim_time, sim_data, label=sim_label)
        axes[0].set_ylabel(sim_label)
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot MMS data
        for i, var in enumerate(mms_tplot_vars):
            pytplot.tplot(var, var_label=pytplot.data_quants[var].name, fig=fig, axis=axes[i+1])
            axes[i+1].grid(True)

        axes[-1].set_xlabel('Time')
        plt.tight_layout()
        plt.show()

    def plot_phase_space(self, f, t, save=False):
        """Plot the phase space distribution."""
        self.ax_phase.clear()
        im = self.ax_phase.pcolormesh(
            self.grid.x, self.grid.v, f.T,
            shading='auto', cmap='viridis'
        )
        self.ax_phase.set_xlabel('x (λ_D)')
        self.ax_phase.set_ylabel('v (v_th)')
        self.ax_phase.set_title(f'Phase Space Distribution (t = {t:.3f})')
        self.fig_phase.colorbar(im, ax=self.ax_phase, label='f(x,v)')
        
        if save:
            plt.savefig(f'plots/phase_space_t{t:.3f}.png')
        else:
            plt.draw()
            plt.pause(0.01)
    
    def plot_moments(self, f, t, save=False):
        """Plot the moments of the distribution function in normalized units."""
        # Compute moments
        density = self.grid.get_density(f)
        velocity = self.grid.get_velocity(f)
        v2 = self.grid.get_velocity_squared(f)
        
        # In normalized units, temperature is just (v2 - v^2)/2
        temperature = (v2 - velocity**2) / 2
        
        # Plot density
        self.ax_density.clear()
        self.ax_density.plot(self.grid.x, density)
        self.ax_density.set_xlabel('x (λ_D)')
        self.ax_density.set_ylabel('n (n_0)')
        self.ax_density.set_title(f'Density (t = {t:.3f})')
        
        # Plot velocity
        self.ax_velocity.clear()
        self.ax_velocity.plot(self.grid.x, velocity)
        self.ax_velocity.set_xlabel('x (λ_D)')
        self.ax_velocity.set_ylabel('v (v_th)')
        self.ax_velocity.set_title(f'Velocity (t = {t:.3f})')
        
        # Plot temperature
        self.ax_temperature.clear()
        self.ax_temperature.plot(self.grid.x, temperature)
        self.ax_temperature.set_xlabel('x (λ_D)')
        self.ax_temperature.set_ylabel('T (T_0)')
        self.ax_temperature.set_title(f'Temperature (t = {t:.3f})')
        
        plt.tight_layout()
        
        if save:
            plt.savefig(f'plots/moments_t{t:.3f}.png')
        else:
            plt.draw()
            plt.pause(0.01)
    
    def create_animation(self, f_history, t_history):
        """Create an animation of the phase space evolution."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        def update(frame):
            ax.clear()
            im = ax.pcolormesh(
                self.grid.x, self.grid.v, f_history[frame].T,
                shading='auto', cmap='viridis'
            )
            ax.set_xlabel('x (λ_D)')
            ax.set_ylabel('v (v_th)')
            ax.set_title(f'Phase Space Distribution (t = {t_history[frame]:.3f})')
            fig.colorbar(im, ax=ax, label='f(x,v)')
            return [im]
        
        anim = FuncAnimation(
            fig, update, frames=len(f_history),
            interval=100, blit=True
        )
        return anim
    
    def save_animation(self, anim):
        """Save the animation to a file."""
        anim.save('plots/phase_space_evolution.mp4', writer='ffmpeg', fps=10)
    
    def close(self):
        """Close all figures."""
        plt.close('all') 