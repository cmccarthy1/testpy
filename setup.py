from distutils.core import setup

#!/usr/bin/env python

from contextlib import contextmanager
from contextlib import redirect_stderr
from distutils import sysconfig
from distutils.command.build import build as default_build
from distutils.command.clean import clean as default_clean
from glob import iglob
import os
import shutil
import subprocess
import sys

import config
from setuptools import Extension
from setuptools import setup
from setuptools.command.build_ext import build_ext as default_build_ext
from setuptools.command.egg_info import egg_info as default_egg_info


__version__ = config.version


def rmrf(path: str):
    """Delete the file tree with ``path`` as its root"""
    if os.path.isdir(path) and not os.path.islink(path):
        shutil.rmtree(path)
    else:
        try:
            os.remove(path)
        except OSError:
            pass


@contextmanager
def cd(newdir):
    """Change the current working directory within the context."""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def build_q_lib():
    def run(command):
        print(command)
        return subprocess.run(command.split()).returncode
    command = (f'gcc {"-g" if config.debug else ""} -Wall -Wextra -D KXVER=3 '
               f'-fwrapv -O3 -I{sysconfig.get_python_inc()} '
               '-fPIC -fno-strict-aliasing -L./lib/ -l:e.o '
               '-I./include/ qtestpy.c -shared -o qtestpy.so')
    if run(command):
        exit(1)


def define_extensions():
    """
    Return cythonized Extension objects. Allows for their definitions to be
    deferred until after Numpy and Cython have been installed.
    """
    import numpy as np
    build_dir = os.environ.get("PWD")
    extra_compile_args=[
        '-O3',
        '-Wall',
        '-Wextra',
        '-Wno-unused-variable',
        '-D CYTHON_TRACE=1' if config.debug else '',
        '-D CYTHON_TRACE_NOGIL=1' if config.debug else '',
    ]
    extra_link_args = ['-Wl,-rpath,' + build_dir + '/src/qtestpy/lib']
    return cythonize_extensions([
        Extension(
            name='qtestpy.adapt',
            sources=['qtestpy/adapt.pyx'],
            include_dirs=['src/qtestpy', 'src/qtestpy/include', np.get_include()],
            library_dirs=['src/qtestpy/lib'],
            libraries=[':e.o'],
            extra_compile_args=[
                *extra_compile_args,
                # https://github.com/cython/cython/issues/2498
                '-D NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION',
                # type punning is used to support GUIDs
                '-Wno-strict-aliasing',
            ],
            extra_link_args = extra_link_args,
        ),
    ])


def cythonize_extensions(extensions):
    """Convert .pyx/.pxd Extensions into regular .c/.cpp Extensions"""
    from Cython.Build import cythonize
    with cd(config.script_dir/'src'):
        cythonized = cythonize(
            extensions,
            language_level=3,
            nthreads=4,
            annotate=config.debug,
            # https://cython.readthedocs.io/en/latest/src/userguide/source_files_and_compilation.html#compiler-directives # noqa: E501
            compiler_directives={
                'binding': True,
                'boundscheck': False,
                'wraparound': False,
                'profile': config.debug and not config.pypy,
                'linetrace': config.debug and not config.pypy,
                'always_allow_keywords': True,
                'embedsignature': True,
                'emit_code_comments': True,
                'initializedcheck': False,
                'nonecheck': False,
                'optimize.use_switch': True,
                # Warns about any variables that are implicitly declared
                # without a cdef declaration
                'warn.undeclared': False,
                'warn.unreachable': True,
                'warn.maybe_uninitialized': False,
                'warn.unused': True,
                'warn.unused_arg': False,
                'warn.unused_result': False,
                'warn.multiple_declarators': True,
            },
        )
    for cy in cythonized:
        cy.sources[0] = 'src/' + cy.sources[0]
    return cythonized


class egg_info(default_egg_info):
    def run(self):
        self.distribution.ext_modules = define_extensions()
        super().run()


class build(default_build):
    def run(self):
        with cd(config.script_dir/'src'/'qtestpy'):
            build_q_lib()
        self.distribution.ext_modules = define_extensions()
        super().run()


class build_ext(default_build_ext):
    """
    build_ext command that supported deferring the definition of the
    Extensions list.
    """
    def run(self):
        if self.inplace:
            with cd(config.script_dir/'src'/'qtestpy'):
                build_q_lib()
        self.distribution.ext_modules = define_extensions()
        cmd_opts = self.distribution.command_options.get('build_ext')
        super().initialize_options()
        self.distribution._set_command_options(self, cmd_opts)
        super().finalize_options()
        super().run()


class clean(default_clean):
    """clean command that cleans some extra files and directories."""
    targets = (iglob(x) for x in (
        'build',
        'dist',
        '.eggs',
        'qtestpy.egg-info',
        'src/qtestpy/*.so',
        'src/qtestpy/build',
    ))

    def run(self):
        with cd(config.script_dir):
            for z in (x for y in self.targets for x in y):
                rmrf(str(z))
        src_dir = config.script_dir/'src'/'qtestpy'
        for f in src_dir.iterdir():
            if f.suffix == '.pyx':
                rmrf(str(f.parent/(f.stem + '.c')))
                rmrf(str(f.parent/(f.stem + '.html')))
        with open(os.devnull, 'w') as devnull:
            with redirect_stderr(devnull):
                # This command has some useless/annoying output that can make
                # it appear like there's an issue when there isn't, so we
                # silence it.
                super().run()
                # If there's a real problem the error will propagate out, so
                # we'll still see it


try:
    subprocess.run('nbstripout --install'.split())
except Exception:
    pass

if 'clean' not in sys.argv:
    setup_requires = [
        'cython',
        'numpy',
    ]
    with open(config.script_dir/'classifiers.txt') as classifiers_file:
        classifiers = [x for x in classifiers_file.read().splitlines()
                       if x and not x.startswith('#')]
else:
    setup_requires = []
    classifiers = []



setup(
  name = 'qtestpy',
  packages = ['qtestpy'],
  version = __version__,
  license='Apache 2',
  description = 'This is a test repo for publishing to conda and pip',
  author = 'Conor McCarthy',
  author_email = 'cmccarthy1@kx.com',
  url = 'https://github.com/cmccarthy1/testpy',
  download_url = 'https://github.com/cmccarthy1/testpy/archive/v_01.tar.gz',
  keywords = ['testing', 'pip', 'installation'],
  install_requires=[
          'cython',
          'numpy',
          'pandas',
          'pyarrow',
      ],
  classifiers=classifiers,
  package_dir={'qtestpy':'src/qtestpy'}
)
