#!/bin/bash

for file in outputs/*.txt; do
  echo "==> $file"
  awk '/^\[RESPONSE\]/ {found=1; next} found' "$file"
  echo ""
done