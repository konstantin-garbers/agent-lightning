#!/bin/sh

# Start the edgebox service in the background
echo "Starting edgebox service..."
edgebox &

# Wait a few seconds for the service to initialize (optional, but good practice)
sleep 3

# Run the main python application in the foreground
# This keeps the container running
echo "Starting multimodal agent..."
python3 examples/multimodal_agent.py