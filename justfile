# Data Analysis Pro - Cross-platform startup script
# Automatically detects OS and runs appropriate startup script

# Default recipe - start the application
start:
    @echo "Starting Data Analysis Pro..."
    @just _detect-os

# Detect OS and run appropriate script
_detect-os:
    #!/usr/bin/env sh
    if [ "$(uname)" = "Darwin" ]; then
        echo "Detected macOS, running shell script..."
        chmod +x scripts/start.sh
        ./scripts/start.sh
    elif [ "$(expr substr $(uname -s) 1 5)" = "Linux" ]; then
        echo "Detected Linux, running shell script..."
        chmod +x scripts/start.sh
        ./scripts/start.sh
    elif [ "$(expr substr $(uname -s) 1 10)" = "MINGW32_NT" ] || [ "$(expr substr $(uname -s) 1 10)" = "MINGW64_NT" ] || [ "$OS" = "Windows_NT" ]; then
        echo "Detected Windows, running batch script..."
        ./scripts/start.bat
    else
        echo "Unknown OS, trying shell script..."
        chmod +x scripts/start.sh
        ./scripts/start.sh
    fi

# Windows-specific recipe
windows:
    @echo "Starting on Windows..."
    @./scripts/start.bat

# Unix-like systems recipe (Linux/macOS)
unix:
    @echo "Starting on Unix-like system..."
    @chmod +x scripts/start.sh
    @./scripts/start.sh

# Show help
help:
    @echo "Data Analysis Pro - Available commands:"
    @echo "  just start    - Auto-detect OS and start application"
    @echo "  just windows  - Force Windows startup"
    @echo "  just unix     - Force Unix/Linux/macOS startup"
    @echo "  just help     - Show this help message"

# Clean up temporary files
clean:
    @echo "Cleaning up temporary files..."
    @rm -rf __pycache__ .pytest_cache *.pyc
    @echo "Clean complete."

# Install dependencies
install:
    @echo "Installing dependencies..."
    @uv sync
    @echo "Dependencies installed."