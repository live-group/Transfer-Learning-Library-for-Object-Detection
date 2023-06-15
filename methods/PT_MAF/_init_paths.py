import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)
project_dir = osp.join(this_dir, '../../')   # go to the root path of this project

# Add lib to PYTHONPATH
lib_path = osp.join(project_dir, 'lib')
add_path(lib_path)

coco_path = osp.join(project_dir, 'data', 'coco', 'PythonAPI')
add_path(coco_path)
