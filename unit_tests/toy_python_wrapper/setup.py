from distutils.core import setup, Extension

audiomodule = Extension('audio',
        sources = ['audiomodule.c', 'audio.cpp']
    )

setup (name = 'audio',
       version = '0.1',
       description = 'audio for ticker',
       ext_modules = [audiomodule])
