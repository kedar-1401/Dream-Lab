import sys

# List of packages to check
packages = ['torch']

for package in packages:
    try:
        __import__(package)
        print(f"{package} is successfully imported")
    except ImportError:
        print(f"Failed to import {package}")

print("Python executable:", sys.executable)
print("Python version:", sys.version)
