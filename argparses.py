# config.py
import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device_id', type=str, default='528e7355', help="device_id")
    parser.add_argument('--y', type=int, default=15, help="An integer value for y")
    return parser.parse_args()

# 解析参数并存储在全局变量中
args = get_args()
