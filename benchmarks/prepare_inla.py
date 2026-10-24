"""Explicitly download INLA into a separately prepared pyINLA environment.

Only run this script when installing the optional native engine is intended.
Normal Astrolabe imports and fits never invoke it or download software.
"""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", required=True)
    parser.parse_args()
    from pyinla.binary import BinaryManager
    manager = BinaryManager()
    launcher = Path(manager.download_binary())
    # New upstream portable archives put the ELF executable in bin/inla.
    # pyINLA 0.1.11 still checks for a sibling inla/inla.mkl. The launcher
    # resolves its own symlink and already locates the packaged libraries.
    legacy = launcher.parent / "inla"
    portable = launcher.parent / "bin" / "inla"
    if not manager.is_installed() and portable.is_file() and not legacy.exists():
        legacy.symlink_to(Path("bin") / "inla")
    if not manager.is_installed():
        raise RuntimeError("Downloaded engine is not recognized by pyINLA; inspect its installation")
    print(launcher)


if __name__ == "__main__":
    main()
