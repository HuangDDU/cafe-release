import os

import ipynbname


def get_img_dir():
    """
    自动获取当前notebook文件名，在img/下创建同名文件夹并返回其路径。
    用法：
        save_img_dir = get_img_dir()
    """
    # notebook_path = ipynbname.path()
    notebook_name = ipynbname.name()  # 获取文件名
    img_dir = os.path.join(os.getcwd(), "img", notebook_name)
    if os.path.exists(img_dir):
        print(f"Image directory already exists: {img_dir}")
    else:
        print(f"Creating image directory: {img_dir}")
        os.makedirs(img_dir)
    return img_dir
