#!/usr/bin/env python3
"""Check Python version and provide installation guidance."""

import sys

def main():
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 14:
        print("\n⚠️  WARNING: You are using Python 3.14+")
        print("\nMany packages (pandas, numpy, scipy) don't have pre-built wheels for Python 3.14 yet.")
        print("\nRecommended solutions:")
        print("\n1. Use Python 3.11 or 3.12 (RECOMMENDED):")
        print("   pyenv install 3.12.0")
        print("   pyenv local 3.12.0")
        print("   python -m venv venv")
        print("   ./venv/bin/pip install -r requirements.txt")
        print("\n2. Build packages from source (SLOW):")
        print("   sudo pacman -S gcc gcc-fortran openblas lapack")
        print("   ./venv/bin/pip install numpy pandas scipy --no-build-isolation")
        print("   ./venv/bin/pip install -r requirements.txt")
        print("\n3. Use conda instead:")
        print("   conda create -n portfolio python=3.12")
        print("   conda activate portfolio")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    elif version.major == 3 and version.minor >= 11:
        print("\n✅ Python version is supported!")
        print("You can proceed with: ./venv/bin/pip install -r requirements.txt")
        sys.exit(0)
    else:
        print("\n⚠️  Python 3.11+ is recommended for best compatibility.")
        sys.exit(0)

if __name__ == "__main__":
    main()
