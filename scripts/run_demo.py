import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import subprocess
import sys


def main() -> None:
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', 'src/demo/app.py'], check=True)


if __name__ == '__main__':
    main()
