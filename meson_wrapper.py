import os
import subprocess
import sys


def call_meson_with(args):
    exit(subprocess.call([f"meson",*args[1:]] if len(args) > 1 else ["meson"], env=os.environ))

if len(sys.argv) < 1:
    call_meson_with([])

if sys.argv[1] != "setup" and len(sys.argv) == 2 or sys.argv[2].startswith("-"):
    call_meson_with(sys.argv)

project_root = sys.argv[2]
builddir = sys.argv[3]

conan_install = f'conan profile detect || (call ) && conan install {project_root} --output-folder={builddir} --build=missing'
os.environ['PKG_CONFIG_PATH'] = f"{builddir}/build/generators:{os.environ.get('PKG_CONFIG_PATH', '')}"

meson_command = f'meson {' '.join(sys.argv[1:])}'

sys.exit(
    subprocess.call(["cmd.exe", "/c", f"{conan_install} && call {builddir}\\build\\generators\\conanbuild.bat && {meson_command}"], env = os.environ)
    if sys.platform == "win32"
    else
    subprocess.call(["bash", "-c", f"{conan_install} && source {builddir}/build/generators/conanbuild.sh && {meson_command}"], env = os.environ)
)