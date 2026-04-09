#!/usr/bin/env bash
# LocalAI Platform — Linux/Mac Installer

set -e
cd "$(dirname "$0")"

echo "=== LocalAI Platform Installer ==="
echo ""

# Check Python 3.10+
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PY_VER=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "Found: Python $PY_VER"

PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)")
if [ "$PY_MINOR" -lt 10 ]; then
    echo "ERROR: Python 3.10+ required (found 3.$PY_MINOR)"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate and install
source .venv/bin/activate
pip install --upgrade pip -q
pip install -r requirements.txt

# Make scripts executable
chmod +x start.sh

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To start LocalAI Platform:"
echo "  ./start.sh"
echo ""
echo "Or directly:"
echo "  .venv/bin/python start.py"
