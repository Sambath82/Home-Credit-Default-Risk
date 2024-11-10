# notebook_setup.py
import sys

# Define root path
RootPath = '/Users/sam/Desktop/GitHub/Project/Home_Credit_Default_Risk/'

# Add the utils directory to the system path
sys.path.append(f'{RootPath}Utils')

# Import configuration
import config

# Expose RootPath and config for easy access in notebooks
__all__ = ['RootPath', 'config']

# Check and print the paths (optional)
#print(f"Root Path: {RootPath}")
#print(f"\nRaw Data Path: {config.RawDataPath}")
