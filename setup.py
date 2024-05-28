import os
from setuptools import find_packages, setup
from pathlib import Path

PACKAGE_NAME = "tinynet"

here = Path(__file__).parent

long_description = (here / "README.md").read_text(encoding="utf-8")

about = {}
exec(
    (here / PACKAGE_NAME.replace(".", os.path.sep) / "__version__.py").read_text(
        encoding="utf-8"
    ),
    about,
)

required = ["click", "tqdm", "streamlit"]
extras_require = {
    "dev": ["pip-tools", "pytest", "python-Levenshtein"],
    "serve": ["uvicorn[standard]", "fastapi", "python-multipart", "pydantic"],
}

entry_points = """
[console_scripts]
tinynet = tinynet.cli:cli
"""

setup(
    name=PACKAGE_NAME,
    version=about["__version__"],
    description="Python3 package for rl/ops",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="leepand",
    author_email="pandeng.li@163.com",
    license="Apache 2.0",
    url="https://github.com/leepand/TinyNet",
    platforms=["Mac", "Linux", "Windows"],
    # package_dir={'':"tinynet"},
    packages=find_packages(),
    include_package_data=True,
    data_files=[
        (
            "",
            [],
        )
    ],
    # entry_points=entry_points,
    install_requires=required,
    extras_require=extras_require,
    zip_safe=False,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python",
        "Programming Language :: Python :: Implementation",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
