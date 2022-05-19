#!/bin/bash
set -eux

# KILL=1 bash run_remote.sh 'cd reactji; pip install -r requirements.txt; LOGLEVEL=INFO PRECISION=32 RUN_NAME=L1_FULL_PRECISION_WARMUP LR=3e-5 BATCH_SZ=128 python train.py'
REMOTE_IP=${REMOTE_IP:-104.171.200.163}

rsync -avz --exclude '*.pth' ./ ubuntu@$REMOTE_IP:~/reactji/
# ssh ubuntu@$REMOTE_IP 'mkdir ~/runs; ln -s ~/runs ~/reactji/runs'
if [[ -n "${KILL:-}" ]]; then
    ssh ubuntu@$REMOTE_IP 'ps auxww | grep python | grep ubuntu | grep -v grep | awk "{print \$2}" | xargs kill'
fi
ssh ubuntu@$REMOTE_IP "$@"
