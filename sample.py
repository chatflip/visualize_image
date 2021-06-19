import os

from visualizer import Visualizer

if __name__ == "__main__":
    src_path = "giraffe.jpg"
    dst_path = "output"
    os.makedirs(dst_path, exist_ok=True)
    vis = Visualizer(src_path, dst_path)
    # vis.show_all()
    vis.save_all()
