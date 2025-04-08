#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="$(git rev-parse --show-toplevel)/scripts/vscode"
DEST_DIR="$(git rev-parse --show-toplevel)/.vscode"

# Check if the git repository root is found
if [ -z "$DEST_DIR" ]; then
    echo "Error: This script must be run within a git repository."
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy the vscode folder to the destination
cp -r "$SOURCE_DIR/"* "$DEST_DIR"

echo "vscode config has been copied to $DEST_DIR"
