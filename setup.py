from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ManipulaPy",
    # Version is canonical in pyproject.toml; this duplicate is kept synced
    # for any legacy tooling that still parses setup.py directly. pip and
    # `python -m build` use pyproject.toml's [project] table, not this.
    version="1.3.2.post1",
    author="Mohamed Aboelnasr",
    author_email="aboelnasr1997@gmail.com",
    description="A comprehensive, GPU-accelerated Python framework for robotic manipulation, perception, and control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boelnasr/ManipulaPy",
    
    # FIXED: Properly find all packages including ManipulaPy_data
    packages=find_packages(include=['ManipulaPy', 'ManipulaPy.*']),
    
    # Legacy setup.py path mirrors pyproject.toml's lightweight core install.
    # Heavy/platform-specific dependencies live in extras below.
    install_requires=[
        "numpy>=2.0,<3.0",
        "scipy>=1.13",
        "matplotlib>=3.9",
        "numba>=0.60",
        "pillow>=8.0.0",
    ],

    # Optional dependencies for enhanced functionality. Keep ultralytics at
    # 8.4.0+ so the vision extra works with the NumPy 2.x runtime range.
    extras_require={
        "minimal": [
            "numpy>=2.0,<3.0",
            "scipy>=1.13",
            "matplotlib>=3.9",
            "numba>=0.60",
            "pybullet>=3.2.5,<4.0",
        ],
        "simulation": [
            "pybullet>=3.2.5,<4.0",
        ],
        "urdf": [
            "trimesh>=3.9.14",
        ],
        "vision": [
            "opencv-python>=4.5",
            "ultralytics>=8.4.0",
            "torch>=1.8.0",
        ],
        "ml": [
            "torch>=1.8.0",
            "scikit-learn>=1.0",
        ],
        # Default GPU extra targets CUDA 12.x (Ubuntu 22.04 + modern
        # NVIDIA apt repos, driver 525+). Pinned to 13.x because CuPy
        # 14.x raised its numpy floor to 2.2 and breaks ultralytics.
        "cuda": [
            "cupy-cuda12x>=13.0,<14.0; sys_platform != 'darwin'",
        ],
        "all": [
            "pybullet>=3.2.5,<4.0",
            "trimesh>=3.9.14",
            "opencv-python>=4.5",
            "ultralytics>=8.4.0",
            "torch>=1.8.0",
            "scikit-learn>=1.0",
            "cupy-cuda12x>=13.0,<14.0; sys_platform != 'darwin'",
        ],
        "gpu-cuda11": [
            "cupy-cuda11x>=13.0,<14.0; sys_platform != 'darwin'",
        ],
        "gpu-cuda12": [
            "cupy-cuda12x>=13.0,<14.0; sys_platform != 'darwin'",
        ],
        "gpu-rocm": [
            "cupy-rocm-4-3>=10.0.0",
        ],
        "gpu-pycuda": [
            "pycuda>=2021.1",
        ],
        "vision-headless": [
            "opencv-python-headless>=4.5",
            "ultralytics>=8.4.0",
            "pillow>=8.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "pytest-mock>=3.6.0",
            "pytest-xvfb>=2.0.0",
            "pytest-timeout>=2.1.0",
            "pytest-benchmark>=4.0.0",
            "coverage[toml]>=6.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "mypy>=0.950",
            "pre-commit>=2.15.0",
            "tox>=4.0.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.15.0",
            "sphinx-autodoc-typehints>=1.12.0",
            "nbsphinx>=0.8.0",
            "jupyter>=1.0.0",
        ],
    },
    
    # Package classification
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
        "Topic :: System :: Hardware :: Hardware Drivers",
    ],
    
    # Python version requirements
    python_requires=">=3.9",
    
    # Legacy package-data path follows MANIFEST.in: URDF/docs only, not meshes.
    include_package_data=True,
    package_data={
        "ManipulaPy": [
            "py.typed",
            "ManipulaPy_data/*.md",
            "ManipulaPy_data/*/*.urdf",
            "ManipulaPy_data/*/*/*.urdf",
            "ManipulaPy_data/*/*/*/*.urdf",
        ],
    },
    
    # Project URLs for PyPI
    project_urls={
        "Homepage": "https://github.com/boelnasr/ManipulaPy",
        "Documentation": "https://manipulapy.readthedocs.io/",
        "Repository": "https://github.com/boelnasr/ManipulaPy.git",
        "Issues": "https://github.com/boelnasr/ManipulaPy/issues",
        "Discussions": "https://github.com/boelnasr/ManipulaPy/discussions",
        "Changelog": "https://github.com/boelnasr/ManipulaPy/blob/main/CHANGELOG.md",
        "Paper": "https://joss.theoj.org/papers/10.21105/joss.xxxxx",  # Update when published
    },
    
    # Keywords for discoverability
    keywords=[
        # Core robotics
        "robotics", "manipulator", "robot-arm", "kinematics", "dynamics",
        "jacobian", "forward-kinematics", "inverse-kinematics", 
        
        # Planning and control
        "trajectory-planning", "path-planning", "motion-planning",
        "control-systems", "pid-control", "computed-torque",
        
        # Simulation and modeling
        "simulation", "pybullet", "physics-simulation", "urdf",
        "robot-modeling", "serial-manipulator",
        
        # Computer vision and perception
        "computer-vision", "perception", "stereo-vision", "yolo",
        "object-detection", "point-cloud", "obstacle-detection",
        
        # Performance and acceleration
        "cuda", "gpu-acceleration", "high-performance", "real-time",
        "parallel-computing", "scientific-computing",
        
        # File formats and standards
        "urdf-parser", "se3", "lie-groups", "screw-theory",
    ],
    
    # Entry points for command-line tools (if any)
    entry_points={
        "console_scripts": [
            # Uncomment and modify if you want CLI tools
            # "manipulapy-benchmark=ManipulaPy.Benchmark.quick_benchmark:main",
            # "manipulapy-viewer=ManipulaPy.tools.urdf_viewer:main",
        ],
    },
    
    # Zip safety
    zip_safe=False,
    
    # Platform compatibility
    platforms=["any"],
)
