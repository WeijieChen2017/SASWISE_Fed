#!/usr/bin/env python3
"""
Runner script for the SASWISE Fed-101 simulation.
This script serves as a direct entry point for running the simulation.
It imports the main function from the simulation module and runs it.
"""

import os
import sys

# Add the package root to Python path
package_root = os.path.dirname(os.path.abspath(__file__))
if package_root not in sys.path:
    sys.path.insert(0, package_root)

try:
    # First try the direct import method
    from saswise_fed_101.simulation import main
    print("Imported simulation using package import")
except ImportError:
    # If that fails, try importing from the relative path
    sys.path.insert(0, os.path.join(package_root, 'saswise_fed_101'))
    from simulation import main
    print("Imported simulation using direct file import")

if __name__ == "__main__":
    print("Starting SASWISE Fed-101 simulation...")
    main() 