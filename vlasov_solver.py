import os
import numpy as np
import traceback
import time
from grid import Grid
from field import ElectricField
from advection import SemiLagrangianAdvection
from diagnostics import Diagnostics

def run_simulation(
    grid,
    field,
    advection,
    diag,
    f,
    nx=64,               # Number of spatial grid points
    nv=64,               # Number of velocity grid points
    x_min=0.0,           # Minimum spatial coordinate (in Debye lengths)
    x_max=20.0,          # Maximum spatial coordinate (in Debye lengths)
    v_min=-4.0,          # Minimum velocity (in thermal velocities)
    v_max=4.0,           # Maximum velocity (in thermal velocities)
    t_end=20.0,          # End time (in plasma periods)
    dt=0.1,              # Initial time step
    plot_interval=1.0,   # Interval between plots
    save_plots=True,     # Whether to save plots
    field_type='self_consistent',  # Change default to self_consistent
    max_steps=10000,     # Maximum number of time steps
    n_threads=None,      # Number of threads for parallel processing
    **field_params       # Additional field parameters
):
    """
    Run the Vlasov simulation with the given parameters.
    All quantities are in normalized units:
    - Length: normalized by Debye length
    - Time: normalized by plasma frequency
    - Velocity: normalized by thermal velocity
    - Electric field: normalized by k_B*T/(e*lambda_D)
    """
    try:
        print("Starting simulation setup...")
        print("Using normalized units:")
        print("- Length: normalized by Debye length")
        print("- Time: normalized by plasma frequency")
        print("- Velocity: normalized by thermal velocity")
        print("- Electric field: normalized by k_B*T/(e*lambda_D)")
        
        # Performance monitoring
        perf_stats = {
            'field_compute_time': [],
            'advection_time': [],
            'total_step_time': [],
            'dt_history': []
        }
        
        # Create output directory
        if save_plots:
            os.makedirs('plots', exist_ok=True)
            print("Created plots directory")
        
        # Initialize grid
        print("\nInitializing grid...")
        print(f"Grid initialized with {nx}x{nv} points")
        print(f"Grid spacing: dx = {grid.dx:.3e}, dv = {grid.dv:.3e}")
        
        # Initialize electric field
        print("\nInitializing electric field...")
        if field_type == 'self_consistent' and 'n0' not in field_params:
            field_params['n0'] = 1.0
        field = ElectricField(grid, field_type=field_type, **field_params)
        print(f"Electric field initialized with type: {field_type}")
        
        # Initialize advection solver with specified number of threads
        print("Initializing advection solver...")
        advection = SemiLagrangianAdvection(grid, n_threads=n_threads)
        print(f"Advection solver initialized with {advection.n_threads} threads")
        
        # Initialize diagnostics
        print("Initializing diagnostics...")
        diag = Diagnostics(grid)
        print("Diagnostics initialized")
        
        # Initialize distribution function
        print("\nInitializing distribution function...")
        f = grid.initialize_maxwellian(
            n0=1.0,
            v0=0.0,
            vt=1.0,
            x0=x_max/2,
            sigma=1.0
        )
        print("Distribution function initialized")
        print(f"Initial total particles: {grid.get_total_particles(f):.6e}")
        
        # Initialize time and history
        t = 0.0
        f_history = [f.copy()]
        t_history = [t]
        e_center_history = []
        center_index = np.argmin(np.abs(grid.x - (grid.x_min + grid.x_max) / 2))
        step = 0
        last_plot_time = 0.0
        
        print("\nStarting main time loop...")
        # Main time loop
        while t < t_end and step < max_steps:
            try:
                step_start_time = time.time()
                step += 1
                
                # Dynamic print interval
                print_interval = max(1, max_steps // 500) * 10
                if step % print_interval == 0:
                    print(f"Step {step}, t = {t:.3f}, dt = {dt:.3e}")
                
                # Compute electric field and acceleration
                field_start = time.time()
                E = field.compute_field(f) if field_type == 'self_consistent' else field.compute_field()
                a = field.get_acceleration(E)
                field_time = time.time() - field_start
                perf_stats['field_compute_time'].append(field_time)
                
                e_center_history.append(E[center_index])
                
                # Compute stable time step
                dt_stable = advection.compute_cfl_dt(a)
                dt = min(dt, dt_stable)
                perf_stats['dt_history'].append(dt)
                
                if dt < 1e-6:
                    print(f"Warning: dt too small ({dt:.2e}), simulation may be unstable")
                    break
                
                # Advect distribution function
                advection_start = time.time()
                f = advection.advect(f, dt, a)
                advection_time = time.time() - advection_start
                perf_stats['advection_time'].append(advection_time)
                
                # Update time
                t += dt
                
                # Performance monitoring
                step_time = time.time() - step_start_time
                perf_stats['total_step_time'].append(step_time)
                
                # Store history and plot if needed
                if t - last_plot_time >= plot_interval:
                    f_history.append(f.copy())
                    t_history.append(t)
                    last_plot_time = t
                    
                    if save_plots:
                        diag.plot_phase_space(f, t, save=True)
                        diag.plot_moments(f, t, save=True)
                    else:
                        diag.plot_phase_space(f, t)
                        diag.plot_moments(f, t)
                    
                    # Print performance stats
                    avg_field_time = np.mean(perf_stats['field_compute_time'][-print_interval:])
                    avg_advection_time = np.mean(perf_stats['advection_time'][-print_interval:])
                    avg_step_time = np.mean(perf_stats['total_step_time'][-print_interval:])
                    print(f"\nPerformance stats (last {print_interval} steps):")
                    print(f"Average field computation time: {avg_field_time:.3e} s")
                    print(f"Average advection time: {avg_advection_time:.3e} s")
                    print(f"Average total step time: {avg_step_time:.3e} s")
                    print(f"Current total particles: {grid.get_total_particles(f):.6e}")
            
            except Exception as e:
                print(f"\nError during time step {step} at t = {t:.3f}:")
                print(traceback.format_exc())
                raise
        
        # Simulation completion
        if step >= max_steps:
            print(f"\nWarning: Reached maximum number of steps ({max_steps})")
        else:
            print("\nSimulation completed successfully!")
        
        print(f"Final time: {t_history[-1]:.3f}")
        print(f"Number of time steps: {step}")
        print(f"Number of saved states: {len(t_history)}")
        
        # Print final performance statistics
        print("\nFinal Performance Statistics:")
        print(f"Average field computation time: {np.mean(perf_stats['field_compute_time']):.3e} s")
        print(f"Average advection time: {np.mean(perf_stats['advection_time']):.3e} s")
        print(f"Average total step time: {np.mean(perf_stats['total_step_time']):.3e} s")
        print(f"Minimum time step: {min(perf_stats['dt_history']):.3e}")
        print(f"Maximum time step: {max(perf_stats['dt_history']):.3e}")
        
        # Create and save animation
        if save_plots and len(f_history) > 1:
            print("\nCreating animation...")
            anim = diag.create_animation(f_history, t_history)
            diag.save_animation(anim)
            print("Animation saved")
        
        # Close figures
        diag.close()
        
        return f_history, t_history, perf_stats
        
    except Exception as e:
        print("\nFatal error in simulation:")
        print(traceback.format_exc())
        raise

if __name__ == '__main__':
    try:
        print("Jumping to animation creation (skipping simulation loop)…")
        nx = 128
        nv = 128
        x_min = 0.0
        x_max = 20.0
        v_min = -4.0
        v_max = 4.0
        grid = Grid(nx, nv, x_min, x_max, v_min, v_max)
        diag = Diagnostics(grid)
        # (Option 1: Simulate a short history (e.g. 10 frames) for demo purposes)
        f = grid.initialize_maxwellian(n0=1.0, v0=0.0, vt=1.0, x0=x_max / 2, sigma=1.0)
        f_history = [f.copy() for _ in range(10)]
        t_history = [i * 0.5 for i in range(10)]
        # (Option 2: Alternatively, you could load a saved history (if available) from a file, e.g. "saved_f_history.npy" and "saved_t_history.npy")
        # (For example, uncomment the following lines if you have saved arrays:)
         # f_history = np.load("saved_f_history.npy")
         # t_history = np.load("saved_t_history.npy")
        print("Creating animation (using "short" history)…")
        anim = diag.create_animation(f_history, t_history)
        diag.save_animation(anim)
        print("Animation (phase_space_evolution.mp4) saved in "plots" directory.")
    except Exception as e:
        print("\nFatal error (animation creation):")
        import traceback
        print(traceback.format_exc())
        raise 