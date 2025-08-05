#!/bin/bash

# Exit on any error
set -e

echo "Starting Stock Analysis API..."

# Check if required environment variables are set
required_vars=("OPENAI_API_KEY" "SEC_API_API_KEY")
for var in "${required_vars[@]}"; do
    if [ -z "${!var}" ]; then
        echo "Warning: $var is not set. Some features may not work."
    fi
done

# Set default port if not specified
export PORT=${PORT:-8000}

echo "Environment variables configured"
echo "Port: $PORT"

# Start the FastAPI application
python app.py