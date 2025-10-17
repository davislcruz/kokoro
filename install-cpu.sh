#!/bin/bash
# Install kokoro with CPU-only PyTorch and locked versions
set -e

echo "Creating Python virtual environment..."
python3 -m venv venv

echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing stable spacy and blis first (avoid compilation issues)..."
pip install --extra-index-url https://download.pytorch.org/whl/cpu torch torchvision
pip install spacy==3.8.7 blis==1.3.0 thinc==8.3.6

echo "Installing misaki from GitHub (avoids PyPI conflicts)..."
pip install git+https://github.com/hexgrad/misaki.git

echo "Installing remaining dependencies..."
pip install -r requirements-cpu.txt

echo "Installing kokoro in development mode..."
pip install -e .

echo "✅ Installation complete!"
echo ""
echo "🎯 Key packages installed:"
echo "  - Python: $(python --version)"
echo "  - PyTorch: CPU-only version"
echo "  - Misaki: 0.9.4 from GitHub"
echo "  - SpaCy: 3.8.7 (stable)"
echo ""
echo "To activate the environment: source venv/bin/activate"
echo "To test kokoro: python -m kokoro --help"