from setuptools import find_packages, setup
from typing import List
from pathlib import Path

HYPEN_E_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """Return requirements from a requirements file.

    - Handles missing/empty files gracefully (returns []).
    - Trims whitespace.
    - Ignores empty lines and comments.
    """

    path = Path(file_path)
    if not path.exists():
        return []

    requirements: List[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        requirements.append(line)

    if HYPEN_E_DOT in requirements:
        requirements = [r for r in requirements if r != HYPEN_E_DOT]

    return requirements


setup(
    name="mlproject",
    version="0.0.1",
    author="Tarunreddy",
    author_email="btarunreddy18@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)

