from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ManipulaPy",
    version="1.1.0",  # keep at 1.1.0
    author="Mohamed Aboelnasr",
    author_email="aboelnasr1997@gmail.com",
    description="A modular, GPU-accelerated Python package for robotic manipulator simulation and control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boelnasr/ManipulaPy",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24,<2.0",
        "scipy>=1.10,<1.13",
        "pybullet>=3.2.5",
        "urchin>=0.0.28",
        "trimesh>=4.0,<4.2",
        "opencv-python>=4.5,<5.0",
        "scikit-learn>=1.3,<1.6",
        "matplotlib>=3.3",
        "ultralytics>=8.0",    # YOLO-based perception
        "torch>=1.8.0",        # required by ultralytics
        "cupy-cuda11x>=10.0.0",# optional CUDA arrays & kernels
    ],
    extras_require={
        "gpu": [
            "pycuda>=2021.1",    # legacy PyCUDA support
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "isort>=5.0.0",
            "mypy>=0.910",
            "pre-commit>=2.15.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
        "all": [
            "ultralytics>=8.0",
            "torch>=1.8.0",
            "cupy-cuda11x>=10.0.0",
            "pycuda>=2021.1",
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        "ManipulaPy": [
            "ManipulaPy_data/ur5/ur5.urdf",
            "ManipulaPy_data/ur5/visual/*.dae",
            "ManipulaPy_data/xarm/xarm6_robot.urdf",
            "ManipulaPy_data/xarm/visual/*.dae",
        ],
    },
    project_urls={
        "Documentation": "https://manipulapy.readthedocs.io/",
        "Source": "https://github.com/boelnasr/ManipulaPy",
        "Tracker": "https://github.com/boelnasr/ManipulaPy/issues",
    },
    keywords=[
        "robotics",
        "kinematics",
        "dynamics",
        "trajectory",
        "control",
        "simulation",
        "pybullet",
        "vision",
        "cuda",
    ],
)
