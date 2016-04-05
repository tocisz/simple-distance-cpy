from distutils.core import setup, Extension

distance_wrap = Extension('_distance_wrap',
                           sources = ['distance_wrap.c'])

setup(name = 'simple-distance-cpy',
      version = '0.1',
      description = 'I want euclidean cdist and I want it fast',
      ext_modules = [distance_wrap],

      url='https://github.com/tocisz/simple-distance-cpy',
      author='Tomasz Cichocki',
      author_email='cichymail at gmail dot com')
