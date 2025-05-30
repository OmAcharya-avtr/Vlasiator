# 1D-1V Vlasov Simulation

A sophisticated implementation of the Vlasov equation solver for simulating plasma dynamics and foreshock waves. This code implements a 1D-1V (one spatial dimension, one velocity dimension) Vlasov simulation using a semi-Lagrangian method, providing a powerful tool for studying collisionless plasma phenomena.

## Overview

The simulation solves the Vlasov equation, which governs the evolution of the distribution function f(x,v,t) in phase space. It uses a semi-Lagrangian method to accurately track the evolution of the plasma distribution while maintaining numerical stability through dynamic time-stepping.

### Key Features

- 1D spatial and 1D velocity phase space simulation
- Semi-Lagrangian method for solving the Vlasov equation
- Self-consistent or prescribed electric field options
- Dynamic time-stepping based on CFL conditions
- Comprehensive diagnostics and visualization tools
- Phase space evolution animation generation
- Normalized units for universal applicability
- Modular code structure for easy extension

## Core Components

### Required Files
- `vlasov_solver.py` - Main simulation driver that orchestrates the entire simulation
- `grid.py` - Handles spatial and velocity grid setup and management
- `field.py` - Manages electric field calculations (self-consistent or prescribed)
- `advection.py` - Implements the semi-Lagrangian advection scheme
- `diagnostics.py` - Handles plotting, diagnostics, and animation generation
- `requirements.txt` - Lists all necessary Python dependencies

### Dependencies
```
numpy (≥1.21.0)
matplotlib (≥3.4.0)
scipy (≥1.7.0)
ffmpeg-python (0.2.0)
pyspedas
numba (≥0.56.0)
```

## Installation

1. Clone the repository to your local machine
2. Navigate to the project directory:
   ```bash
   cd /path/to/Vlasliator
   ```
3. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```
   
   Note: If you encounter NumPy compatibility issues (e.g., "A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6..."), you may need to downgrade NumPy:
   ```bash
   pip3 install -r requirements.txt numpy<2.0.0
   ```

## Usage

### Running the Simulation

1. Execute the main script:
   ```bash
   python3 vlasov_solver.py
   ```

   Note: The current version is configured to "jump" directly to animation creation using a demo history. For full simulation mode, you'll need to modify the main block in `vlasov_solver.py`.

2. To stop the simulation (if running in background):
   ```bash
   pkill -f "python3 vlasov_solver.py"
   ```

### Simulation Parameters

The simulation can be configured with various parameters:
- Grid resolution (nx, nv)
- Spatial domain (x_min, x_max)
- Velocity domain (v_min, v_max)
- Simulation duration (t_end)
- Time step (dt) - automatically adjusted for stability
- Field type (self_consistent or prescribed)
- Number of threads for parallel processing
- Plotting intervals and options

### Output

- The simulation generates diagnostic plots in the `plots/` directory (created automatically if `save_plots=True`)
- A phase space evolution animation is saved as `plots/phase_space_evolution.mp4`
- You can view the animation using any video player or ffplay (if ffmpeg is installed)

## Technical Details

### Normalization
The simulation uses normalized units for universal applicability:
- Length: normalized by Debye length
- Time: normalized by plasma frequency
- Velocity: normalized by thermal velocity
- Electric field: normalized by k_BT/(eλ_D)

### Simulation Process
1. The electric field is computed (self-consistent or prescribed)
2. Particle acceleration is calculated
3. Distribution function is advected in phase space
4. Diagnostics are generated at specified intervals
5. Animation is created from the simulation history

## Future Extensions

- Coupling with Maxwell's equations
- Multiple particle species support
- Spacecraft boundary conditions (MMS data input)
- More sophisticated field solvers
- Enhanced visualization tools
- Performance optimizations

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

[Add your license information here] 