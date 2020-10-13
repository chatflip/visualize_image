import os

from visualizer import Visualizer

if __name__ == '__main__':
    src_path = 'sample.jpg'
    dst_path = 'output'
    os.makedirs(dst_path, exist_ok=True)
    vis = Visualizer(src_path, dst_path)
    vis.save_all()
