#!/bin/bash
# Monitor 4×B200 training in real-time

watch -n 2 '
echo "=== GPU Utilization ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits

echo ""
echo "=== Training Processes ==="
ps aux | grep train_ppo.py | grep -v grep | head -4

echo ""
echo "=== Latest Log (last 10 lines) ==="
tail -10 runs/*.log 2>/dev/null | tail -10
'