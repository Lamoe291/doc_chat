import sys
from pathlib import Path

from doc_chat.cli import run


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m doc_chat <pdf>")
        sys.exit(1)

    run(Path(sys.argv[1]))


if __name__ == "__main__":
    main()