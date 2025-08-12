project('window_capture', 'c', 'cpp',
  version : '0.1.0',
  default_options : ['cpp_std=c++17']
)

fs = import('fs')

src = files('src/window_capture.cpp')

py = import('python')
python_dep = py.find_installation()

obs_game_capture_dir = meson.project_build_root() / 'obs-game-capture-lib'
obs_game_capture_dir = obs_game_capture_dir.replace(meson.project_build_root() / '', './')

if not fs.exists(obs_game_capture_dir)
    run_command('git', 'clone', 'https://github.com/bhargavh/obs-game-capture-lib.git', obs_game_capture_dir, check: true, capture:true)
    run_command('python', '-c', 'import shutil; shutil.rmtree("' + obs_game_capture_dir.replace('./', '') + '/.git", ignore_errors=True)', check: true, capture:true)
endif

opencv_dep = dependency('opencv', method: 'pkg-config', required: true, static: true, native: false)

obs_game_capture_src = files(obs_game_capture_dir / 'game_capture.cpp')

# Get numpy include dir at build time
numpy_include_dir = run_command(python_dep, '-c', 'import numpy; print(numpy.get_include())', check: true, capture: true).stdout().strip()

shared_lib = python_dep.extension_module(
  'window_capture',
  src + obs_game_capture_src,
  dependencies: [opencv_dep],
  include_directories: [include_directories(obs_game_capture_dir), include_directories(numpy_include_dir)],
  install: true,
  install_dir: get_option('libdir'),
  build_by_default: true,
)