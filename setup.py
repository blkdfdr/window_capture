from setuptools import setup, Extension, find_packages
import numpy
import os
import subprocess
import sys
from setuptools.command.build_ext import build_ext as build_ext_orig

# Auto-clone obs-game-capture-lib if not present
if not os.path.exists('obs-game-capture-lib'):
    print('Cloning obs-game-capture-lib...')
    subprocess.check_call([
        'git', 'clone', 'https://github.com/bhargavh/obs-game-capture-lib.git', 'obs-game-capture-lib'
    ])

# Auto-clone vcpkg if not present
if not os.path.exists('vcpkg'):
    print('Cloning vcpkg...')
    subprocess.check_call([
        'git', 'clone', 'https://github.com/microsoft/vcpkg.git', 'vcpkg', '--depth', '1'
    ])

# Bootstrap vcpkg if not already bootstrapped
if os.name == 'nt':
    vcpkg_bootstrap = os.path.join('vcpkg', 'bootstrap-vcpkg.bat')
else:
    vcpkg_bootstrap = os.path.join('vcpkg', 'bootstrap-vcpkg.sh')
if not os.path.exists(os.path.join('vcpkg', 'vcpkg')) and not os.path.exists(os.path.join('vcpkg', 'vcpkg.exe')):
    print('Bootstrapping vcpkg...')
    subprocess.check_call([vcpkg_bootstrap], shell=True, stdout=sys.stdout, stderr=sys.stderr)

class CMakeBuildExt(build_ext_orig):
    def run(self):
        # Set NUMPY_INCLUDE_DIR environment variable for CMake
        numpy_include = numpy.get_include()
        os.environ['NUMPY_INCLUDE_DIR'] = numpy_include

        # Configure and build with CMake
        build_temp = os.path.abspath(self.build_temp)
        build_lib = os.path.abspath(self.build_lib)
        os.makedirs(build_temp, exist_ok=True)
        vcpkg_toolchain = os.path.abspath(os.path.join('vcpkg', 'scripts', 'buildsystems', 'vcpkg.cmake'))
        cmake_args = [
            'cmake',
            '-S', '.',
            '-B', build_temp,
            f'-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={build_lib}',
            f'-DCMAKE_TOOLCHAIN_FILE={vcpkg_toolchain}',
        ]
        if sys.platform == 'win32':
            cmake_args += ['-A', 'x64']
        print('Configuring with CMake:', ' '.join(cmake_args))
        subprocess.check_call(cmake_args)
        build_args = ['cmake', '--build', build_temp, '--config', 'Release']
        print('Building with CMake:', ' '.join(build_args))
        subprocess.check_call(build_args)

setup(
    name="window_capture",
    version="0.1.0",
    ext_modules=[Extension('window_capture', sources=[])],  # sources handled by CMake
    cmdclass={'build_ext': CMakeBuildExt},
    packages=find_packages(),
    include_package_data=True,
    package_data={
        '': ['CMakeLists.txt'],
        'src': ['*.cpp', '*.h'],
    },
)