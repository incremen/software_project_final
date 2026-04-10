from setuptools import setup, Extension

# Add your util file to the sources array
module = Extension(
    'symnmf', 
    sources=['symnmfmodule.c', 'symnmf.c', 'matrix_utils.c'] 
)

setup(
    name='symnmf',
    version='1.0',
    description='SymNMF C Extension for Python',
    ext_modules=[module]
)