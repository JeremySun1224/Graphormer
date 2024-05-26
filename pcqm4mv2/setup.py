# -*- coding: utf-8 -*-
# -*- author: jeremysun1224 -*-

from setuptools import setup, Extension
import numpy as np

# 编译Cython代码
ext_modules = [
    Extension(
        'algos',  # 替换成您的Cython模块名称
        sources=['algos.pyx'],  # 替换成您的Cython源文件
        include_dirs=[np.get_include()],  # 获取NumPy的包含路径
        # 其他编译选项和依赖项
    )
]

setup(
    ext_modules=ext_modules,
    # 其他设置项
)
