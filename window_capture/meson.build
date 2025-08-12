obs_game_capture_dir = meson.project_build_root() / 'obs-game-capture-lib'
obs_game_capture_dir = obs_game_capture_dir.replace(meson.project_build_root() / '', './')

if not fs.exists(obs_game_capture_dir)
    run_command('git', 'clone', 'https://github.com/bhargavh/obs-game-capture-lib.git', obs_game_capture_dir, check: true, capture:true)
    run_command('python', '-c', 'import shutil; shutil.rmtree("' + obs_game_capture_dir.replace('./', '') + '/.git", ignore_errors=True)', check: true, capture:true)
endif
obs_game_capture_src = files(obs_game_capture_dir / 'game_capture.cpp')

opencv_dep = dependency('opencv', method: 'pkg-config', required: true, static: true, native: false)

# Get numpy include dir at build time
numpy_include_dir = run_command(python, '-c', 'import numpy; print(numpy.get_include())', check: true, capture: true).stdout().strip()

src = files('window_capture.cpp')

shared_lib = python.extension_module(
  'window_capture',
  src + obs_game_capture_src,
  install: true,
  dependencies: [opencv_dep],
  include_directories: [include_directories(obs_game_capture_dir), include_directories(numpy_include_dir)],
  build_by_default: true,
  subdir: 'window_capture'
)

python_sources = [
  '__init__.py'
]

python.install_sources(
  python_sources,
  subdir: 'window_capture',
  pure: false
)