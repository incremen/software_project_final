"""Build script for the symnmf Python C extension."""
from setuptools import Extension, setup

symnmf_module = Extension(
    'symnmf',
    sources=['symnmfmodule.c', 'symnmf.c', 'matrix_utils.c'],
)

setup(
    name='symnmf',
    version='1.0',
    description='SymNMF C extension for Python.',
    ext_modules=[symnmf_module],
)
