from distutils.core import setup, Extension

import numpy.distutils.misc_util as npy_util
npy_info = npy_util.get_info('npymath')
distance_wrap = Extension('_distance_wrap', sources = ['distance_wrap.c'], **npy_info)

setup(name = 'simple-distance-cpy',
      version = '0.21',
      description = 'I want euclidean cdist and I want it now',
      py_modules = ['distance'],
      ext_modules = [distance_wrap],

      url='https://github.com/tocisz/simple-distance-cpy',
      author='Tomasz Cichocki',
      author_email='cichymail at gmail dot com')
