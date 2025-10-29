"""
Setup script for building C++ accelerated option pricing modules

Requirements:
- pybind11
- C++11 compatible compiler
- OpenMP (optional, for parallel processing)

Build instructions:
    pip install pybind11
    python setup.py build_ext --inplace

Or install as package:
    pip install .
"""

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
import pybind11

__version__ = '0.1.0'


class get_pybind_include(object):
    """Helper class to determine the pybind11 include path"""

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


ext_modules = [
    Extension(
        'heston_cpp',
        ['heston_bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3', '-march=native', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
    Extension(
        'sabr_cpp',
        ['sabr_bindings.cpp'],
        include_dirs=[
            get_pybind_include(),
            get_pybind_include(user=True)
        ],
        language='c++',
        extra_compile_args=['-std=c++11', '-O3', '-march=native', '-fopenmp'],
        extra_link_args=['-fopenmp'],
    ),
]


# cf http://bugs.python.org/issue26689
def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on
    the specified compiler.
    """
    import tempfile
    import os
    with tempfile.NamedTemporaryFile('w', suffix='.cpp', delete=False) as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        fname = f.name
    try:
        compiler.compile([fname], extra_postargs=[flagname])
    except setuptools.distutils.errors.CompileError:
        return False
    finally:
        try:
            os.remove(fname)
        except OSError:
            pass
    return True


def cpp_flag(compiler):
    """Return the -std=c++[11/14/17] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""
    c_opts = {
        'msvc': ['/EHsc', '/openmp'],
        'unix': ['-O3', '-march=native'],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++', '-mmacosx-version-min=10.14']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts
        # Note: OpenMP on macOS requires libomp
        # Install with: brew install libomp
        c_opts['unix'] += ['-Xpreprocessor', '-fopenmp']
        l_opts['unix'] += ['-lomp']

    def build_extensions(self):
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])

        if ct == 'unix':
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')

        for ext in self.extensions:
            ext.define_macros = [('VERSION_INFO', '"{}"'.format(self.distribution.get_version()))]
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts

        build_ext.build_extensions(self)


setup(
    name='option_pricing_cpp',
    version=__version__,
    author='Quant Developer',
    author_email='',
    url='',
    description='C++ accelerated option pricing modules',
    long_description='High-performance option pricing using C++ with Python bindings',
    ext_modules=ext_modules,
    setup_requires=['pybind11>=2.5.0'],
    cmdclass={'build_ext': BuildExt},
    zip_safe=False,
    python_requires='>=3.7',
)
