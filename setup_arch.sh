#!/bin/bash
# Setup script for Arch Linux with proper Python 3.14 handling

set -e

echo "=========================================="
echo "Portfolio Optimizer - Arch Linux Setup"
echo "=========================================="

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | cut -d' ' -f2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d'.' -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d'.' -f2)

echo "Detected Python: $PYTHON_VERSION"

# Check if using Python 3.14+
if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 14 ]; then
    echo ""
    echo "⚠️  WARNING: Python 3.14 detected!"
    echo "Many packages don't have pre-built wheels for Python 3.14."
    echo ""
    echo "Options:"
    echo "1. Continue with source compilation (slow, ~10-20 min)"
    echo "2. Use pyenv to install Python 3.12 (recommended)"
    echo "3. Exit and manually install Python 3.12"
    echo ""
    read -p "Choose option (1/2/3): " choice
    
    case $choice in
        1)
            echo "Installing build dependencies..."
            sudo pacman -S --needed --noconfirm gcc gcc-fortran openblas lapack
            
            echo "Creating virtual environment..."
            python -m venv venv
            
            echo "Upgrading pip and build tools..."
            ./venv/bin/pip install --upgrade pip wheel setuptools cython
            
            echo "Installing numpy (compilation required)..."
            ./venv/bin/pip install numpy --no-build-isolation
            
            echo "Installing pandas (compilation required)..."
            ./venv/bin/pip install pandas --no-build-isolation
            
            echo "Installing scipy (compilation required)..."
            ./venv/bin/pip install scipy --no-build-isolation
            
            echo "Installing remaining packages..."
            ./venv/bin/pip install -r requirements.txt
            ;;
        2)
            echo "Installing pyenv..."
            if ! command -v pyenv &> /dev/null; then
                sudo pacman -S --needed --noconfirm pyenv
            fi
            
            echo "Installing Python 3.12.0 via pyenv..."
            pyenv install 3.12.0
            pyenv local 3.12.0
            
            echo "Creating virtual environment with Python 3.12..."
            $(pyenv which python) -m venv venv
            
            echo "Installing dependencies..."
            ./venv/bin/pip install --upgrade pip
            ./venv/bin/pip install -r requirements.txt
            ;;
        3)
            echo "Exiting. Please install Python 3.11 or 3.12 and try again."
            exit 0
            ;;
        *)
            echo "Invalid option"
            exit 1
            ;;
    esac
else
    # Python 3.11 or 3.12 - normal installation
    echo "✅ Python version is compatible"
    
    echo "Creating virtual environment..."
    python -m venv venv
    
    echo "Installing dependencies..."
    ./venv/bin/pip install --upgrade pip
    ./venv/bin/pip install -r requirements.txt
fi

# Setup environment file
echo ""
echo "Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your API keys."
fi

# Initialize database
echo ""
echo "Initializing database..."
./venv/bin/python -c "
from app import app
with app.app_context():
    from models.database import db
    db.create_all()
    print('Database initialized!')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To run the application:"
echo "  ./venv/bin/python app.py"
echo ""
echo "Or with flask CLI:"
echo "  ./venv/bin/python -m flask --app app run --host=0.0.0.0 --port=5000 --debug"
echo ""
echo "Don't forget to edit .env with your DeepSeek API key!"
