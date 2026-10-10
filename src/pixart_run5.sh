#!/bin/bash
cd /mnt/data/certimmune/src
export HF_HOME=/mnt/data/hf_cache
while pgrep -f 'pixart_run4.sh' > /dev/null; do sleep 60; done
../venv/bin/python dense2.py
