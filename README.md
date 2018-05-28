# NetworkInterdiction
A set of python routines for solving Network Interdiction Problems using the __Python Optimization Modeling Object__ (__PYOMO__) package for the definition and solving of optimization problems.

Users will need to install their own solvers and ensure accessibility by Pyomo (generally accomplished if accessible via command line interface). Otherwise, users will need to set the `SolverManager` parameter to __NEOS__, a free, web-accessible hosted set of solvers (paramter access forthcoming).

Sample code and sample `csv` files were extracted from PyomoGallery (link forthcoming).

Each interdiction model contains two files:
  1. A `network_interdiction.py` file that has the model defined as a class (reuse as is)
	2. A `NetworkInterdiction.py` file that takes the class and runs the model (copy, modify, rerun)

Future models and implementations will be added as they become available

