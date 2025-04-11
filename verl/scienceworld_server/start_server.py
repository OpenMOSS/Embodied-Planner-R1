import subprocess
import argparse
import os
import time
import signal
import sys

# 保存所有子进程
processes = []

def signal_handler(sig, frame):
    """处理中断信号，确保关闭所有子进程"""
    print("\n接收到中断信号，正在关闭所有服务器...")
    for p in processes:
        if p.poll() is None:  # 如果进程仍在运行
            p.terminate()
    
    print("等待所有服务器关闭...")
    time.sleep(2)
    print("已关闭所有服务器")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="启动多个ALFWorld服务器")
    parser.add_argument("--num_servers", type=int, default=4, help="要启动的服务器数量")
    parser.add_argument("--start_port", type=int, default=8000, help="起始端口号")
    args = parser.parse_args()
    
    # 注册信号处理
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 启动多个服务器
    for i in range(args.num_servers):
        port = args.start_port + i
        server_id = f"server_{i}"
        
        # 构建命令
        cmd = [
            "python", 
            "server.py",  # 假设主服务器文件名为main.py
            "--port", str(port),
            "--server_id", server_id
        ]
        
        # 启动进程
        print(f"启动服务器 {server_id} 在端口 {port}")
        p = subprocess.Popen(cmd)
        processes.append(p)
        
        # 等待短暂时间，避免启动冲突
        time.sleep(1)
    
    print(f"已启动 {args.num_servers} 个服务器实例")
    print(f"服务器端口范围: {args.start_port} - {args.start_port + args.num_servers - 1}")
    print("按 Ctrl+C 停止所有服务器")
    
    # 等待所有进程
    try:
        for p in processes:
            p.wait()
    except KeyboardInterrupt:
        # 处理KeyboardInterrupt，虽然signal_handler也会处理SIGINT
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main()