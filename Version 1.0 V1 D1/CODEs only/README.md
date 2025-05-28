# 1D-1V Vlasov Simulation

A simplified implementation of the Vlasov equation solver for simulating low-frequency foreshock waves. This code implements a 1D-1V (one spatial dimension, one velocity dimension) Vlasov simulation using a semi-Lagrangian method.

## Features

- 1D spatial and 1D velocity phase space simulation
- Semi-Lagrangian method for solving the Vlasov equation
- Simple electrostatic field implementation
- Phase space visualization and diagnostics
- Modular code structure for easy extension

## Requirements

- Python 3.7+
- NumPy
- Matplotlib
- SciPy

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

Run the main simulation:
```bash
python vlasov_solver.py
```

## Project Structure

- `vlasov_solver.py`: Main simulation code
- `grid.py`: Grid initialization and management
- `field.py`: Electric field calculations
- `advection.py`: Velocity advection using semi-Lagrangian method
- `diagnostics.py`: Plotting and diagnostics

## Future Extensions

- Coupling with Maxwell's equations
- Multiple particle species
- Spacecraft boundary conditions (MMS data input)
- More sophisticated field solvers 