requiered_mem=$1

# 获取第一个 GPU 的内存使用情况
free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0| tail -n 1| grep -o '[0-9]*')
# 如果内存使用情况小于设定的阈值（例如 10000 MB），则等待
while [ $free_mem -lt $requiered_mem ] 
do
  echo "GPU memory is low, waiting..."
  sleep 120 # 等待 120 秒
  # 再次检查 GPU 内存剩余是否足够
  free_mem=$(nvidia-smi --query-gpu=memory.free --format=csv -i 0| tail -n 1| grep -o '[0-9]*')
done
  # 如果内存使用情况大于或等于阈值，则运行 python 程序
  echo "GPU memory is enough, running python program..."
