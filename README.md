# Congestion Management - Quantum
Quantum computing exploration for finding optimal solutions for congestion management in electricity transport networks

## Problem
Congestion management is a complex problem for electricity networks with many power plants, transformers and lines.
It can be modelled as a MILP and solved on classical computers. However, the execution time can increase exponentially.
This often leads to simplification of the original problem taking strong hypotheses in order to keep the execution time reasonable.
On the other hand, these simplifications then yield suboptimals solutions.


## Quantum exploration
In this example, we **reformulate the congestion management problem as a QUBO**, opening the way then to using quantum computers.
Note however that here we present a trivial congestion management, with two powerplants, one transformer and two lines.

This first version uses **DWave solvers** to assess the potential of quantum computing for congestion management


