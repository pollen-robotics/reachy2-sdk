#!/bin/bash

# List of VSCode extensions to install
EXTENSIONS=(
    "ms-python.python"
    "ms-python.mypy-type-checker"
    "ms-python.black-formatter"
    "ms-python.flake8"
    "ms-python.isort"
)

# Installing each extension
for EXT in "${EXTENSIONS[@]}"; do
    code --install-extension "$EXT"
done

# Uninstall Pylance if it is installed
if code --list-extensions | grep -q 'ms-python.vscode-pylance'; then
    echo "Uninstalling Pylance..."
    code --uninstall-extension ms-python.vscode-pylance
fi

echo "Extensions installation complete."
