#!/bin/bash
# Build script for mraa from source for Python 3.11
# Run this script with: bash build_mraa.sh

set -e

echo "=== Building mraa from source ==="
echo ""

# Navigate to mraa source directory
MRAA_DIR="/tmp/mraa"
if [ ! -d "$MRAA_DIR" ]; then
    echo "mraa source not found at $MRAA_DIR"
    echo "Cloning mraa repository..."
    cd /tmp
    git clone https://github.com/eclipse/mraa.git
fi

cd "$MRAA_DIR"

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "Error: CMakeLists.txt not found."
    exit 1
fi

# Install build dependencies (requires sudo)
echo "Step 1: Installing build dependencies..."
sudo apt-get install -y build-essential cmake python3-dev python3-setuptools git pkg-config swig

# Create build directory
echo ""
echo "Step 2: Creating build directory..."
mkdir -p build
cd build

# Configure with CMake (Python bindings are enabled by default via SWIG)
echo ""
echo "Step 3: Configuring build with CMake (Python 3.11 bindings will be built)..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build the library and Python bindings
echo ""
echo "Step 4: Building mraa library and Python bindings..."
make -j$(nproc)

# Install the library and Python bindings
echo ""
echo "Step 5: Installing mraa library and Python bindings..."
sudo make install

# Update library cache
echo ""
echo "Step 6: Updating library cache..."
sudo ldconfig

# Test installation
echo ""
echo "Step 7: Testing installation..."
python3 -c "import mraa; print('✓ mraa imported successfully!'); print('Version:', mraa.getVersion())" || {
    echo "Warning: mraa import failed, but library may still be installed"
    echo "You may need to check the installation or rebuild Python bindings"
}

echo ""
echo "=== Build complete! ==="

