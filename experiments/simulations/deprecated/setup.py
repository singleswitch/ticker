from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_modules = [Extension("ticker_scores", ["ticker_scores.pyx"])]
 
for e in ext_modules:
    e.pyrex_directives = {"boundscheck": False, "wraparound":False, "nonecheck":False}
    
setup(
    name = "scores",
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules
)


#python setup.py build_ext --inplace