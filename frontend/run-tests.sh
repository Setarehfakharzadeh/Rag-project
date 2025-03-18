#!/bin/bash

echo "Installing dependencies..."
npm install

echo "Running tests..."
npm test -- --watchAll=false

# Return the exit code from the tests
exit $? 