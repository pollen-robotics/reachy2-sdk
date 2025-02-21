#!/bin/bash

# Define the source and destination directories
SOURCE_DIR="$(git rev-parse --show-toplevel)/scripts/git_hooks"
DEST_DIR="$(git rev-parse --show-toplevel)/.git/hooks"

# Check if the git repository root is found
if [ -z "$DEST_DIR" ]; then
    echo "Error: This script must be run within a git repository."
    exit 1
fi

# Create the destination directory if it doesn't exist
mkdir -p "$DEST_DIR"

# Copy the git_hooks folder to the destination
cp -r "$SOURCE_DIR/"* "$DEST_DIR"

# Make scripts executable
chmod +x "$DEST_DIR"/*

echo "git hooks copied in $DEST_DIR"
