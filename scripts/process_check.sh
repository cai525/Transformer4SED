#!/bin/bash
process_name=$1

# 检查进程是否存在
process_exists=$(ps -aux | grep $process_name | grep -v grep)

# 如果进程存在，则等待
while [ -n "$process_exists" ]
do
  echo "Process $process_name is running, waiting..."
  sleep 120 # 等待 120 秒
  # 再次检查进程是否存在
  process_exists=$(ps -aux | grep $process_name | grep -v grep)
done

# 如果进程不存在，输出提示信息后退出
echo "Process $process_name is not running, exiting..."
