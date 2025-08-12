# run_conan.py
import sys
from conan.cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
