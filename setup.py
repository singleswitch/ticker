from distutils.core import setup, Extension
import os

operating_system = 'Linux' 
is_support = False

if operating_system =='Linux':
    fmod_env_var_name = "TICKER_FMOD_ROOT_DIR"
    if fmod_env_var_name not in os.environ:
        print("WARNING: TICKER_FMOD_ROOT_DIR is not set. It will be "
              "assumed that FMOD libraries and headers reside in the "
              "hard-coded paths in setup.py")
        libfmodex = os.sep.join(['','usr','lib','libfmodex.so'])
        inc_dirs = [ os.sep.join(['','usr','local','include','fmodex'])]  #directory where fmod.hpp is
    else:
        fmod_root_path = os.environ[fmod_env_var_name]
        libfmodex = os.sep.join([fmod_root_path,'api','lib','libfmodex64.so'])
        inc_dirs = [ os.sep.join([fmod_root_path,'api','inc'])]  #directory where fmod.hpp is

    print "Attempting build with the following FMOD configuration:"
    print "    include dir ", inc_dirs
    print "    lib fmodex = ", libfmodex
    audiomodule = Extension('audio',
                        sources = ['audiomodule.c', 'audio.cpp','fmod_alphabet_player.cpp'],
                        include_dirs = inc_dirs,
                        extra_objects =[libfmodex])
    is_support = True
elif operating_system == 'Windows':
    lib_dirs = [ os.sep.join(['C:','Program Files','FMOD SoundSystem', 'FMOD Programmers API Win32','api','lib']) ]
    libs = ['fmodex_vc']
    inc_dirs = [ os.sep.join(['C:','Program Files','FMOD SoundSystem', 'FMOD Programmers API Win32','api','inc']) ]
    audiomodule = Extension('audio',
                        sources = ['audiomodule.c', 'audio.cpp','fmod_alphabet_player.cpp'],
                        include_dirs = inc_dirs,
                        library_dirs = lib_dirs,
                        libraries = libs,
                        define_macros = [("WIN32", None)],                          
                        extra_compile_args = ['/EHsc'])
    is_support = True
if is_support:
    setup(name = 'audio',
        version = '0.1',
        description = 'audio for ticker', 
        ext_modules = [audiomodule])
    print("Remember to either run 'python setup.py install', or manually add "
          "the directory path of built audio.so module to your PYTHONPATH")
else:
    print "Your operating system is not currently supported in this setup file"
    
    
