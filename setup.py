from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ManipulaPy",
    version="1.1.0",  # Major feature release with GPU acceleration and vision pipeline
    author="Mohamed Aboelnasr",
    author_email="aboelnasr1997@gmail.com",
    description="A comprehensive, GPU-accelerated Python framework for robotic manipulation, perception, and control",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boelnasr/ManipulaPy",
    packages=find_packages(),
    
    # Core dependencies - required for basic functionality
    install_requires=[
        "numpy>=1.20.0,<2.0",         # Core numerical operations
        "scipy>=1.7.0,<2.0",          # Scientific computing
        "matplotlib>=3.3.0,<4.0",     # Plotting and visualization
        "pybullet>=3.2.5,<4.0",       # Physics simulation
        "urchin>=0.0.27,<1.0",        # URDF processing
        "trimesh>=3.15.0,<5.0",       # 3D mesh processing
        "opencv-python>=4.5.0,<5.0",  # Computer vision
        "scikit-learn>=1.0.0,<2.0",   # Machine learning (clustering)
        "numba>=0.56.0,<1.0",         # JIT compilation for performance
    ],
    
    # Optional dependencies for enhanced functionality
    extras_require={
        # GPU acceleration with CUDA 11.x
        "gpu-cuda11": [
            "cupy-cuda11x>=10.0.0,<13.0",  # CUDA arrays and kernels
        ],
        
        # GPU acceleration with CUDA 12.x  
        "gpu-cuda12": [
            "cupy-cuda12x>=12.0.0,<13.0",  # CUDA arrays and kernels
        ],
        
        # Legacy GPU support
        "gpu": [
            "cupy-cuda11x>=10.0.0,<13.0",  # Default to CUDA 11.x
            "pycuda>=2021.1,<2025.0",      # Legacy PyCUDA support
        ],
        
        # Vision and perception capabilities
        "vision": [
            "ultralytics>=8.0.0,<9.0",     # YOLO object detection
            "torch>=1.8.0,<3.0",           # Required by ultralytics
            "torchvision>=0.9.0,<1.0",     # Computer vision models
        ],
        
        # Development and testing tools
        "dev": [
            "pytest>=7.0.0,<8.0",
            "pytest-cov>=4.0.0,<5.0", 
            "pytest-benchmark>=4.0.0,<5.0",
            "black>=22.0.0,<24.0",
            "flake8>=5.0.0,<7.0",
            "isort>=5.10.0,<6.0",
            "mypy>=0.990,<2.0",
            "pre-commit>=2.20.0,<4.0",
        ],
        
        # Documentation generation
        "docs": [
            "sphinx>=5.0.0,<8.0",
            "sphinx-rtd-theme>=1.0.0,<3.0",
            "myst-parser>=0.18.0,<3.0",
            "sphinx-autodoc-typehints>=1.19.0,<2.0",
        ],
        
        # Benchmarking and performance analysis
        "benchmark": [
            "psutil>=5.8.0,<6.0",          # System monitoring
            "tabulate>=0.9.0,<1.0",        # Table formatting
            "seaborn>=0.11.0,<1.0",        # Statistical plotting
            "memory-profiler>=0.60.0,<1.0", # Memory usage analysis
        ],
        
        # Complete installation with all features
        "all": [
            # GPU support (CUDA 12.x preferred)
            "cupy-cuda12x>=12.0.0,<13.0",
            "pycuda>=2021.1,<2025.0",
            
            # Vision capabilities
            "ultralytics>=8.0.0,<9.0",
            "torch>=1.8.0,<3.0",
            "torchvision>=0.9.0,<1.0",
            
            # Development tools
            "pytest>=7.0.0,<8.0",
            "pytest-cov>=4.0.0,<5.0",
            "black>=22.0.0,<24.0",
            "flake8>=5.0.0,<7.0",
            
            # Documentation
            "sphinx>=5.0.0,<8.0",
            "sphinx-rtd-theme>=1.0.0,<3.0",
            
            # Benchmarking
            "psutil>=5.8.0,<6.0",
            "tabulate>=0.9.0,<1.0",
            "seaborn>=0.11.0,<1.0",
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
        "Programming Language :: Python :: 3.8",
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
    python_requires=">=3.8",
    
    # Include additional package data
    include_package_data=True,
    package_data={
        "ManipulaPy": [
            # Robot models and configurations
            "ManipulaPy_data/ur5/*.urdf",
            "ManipulaPy_data/ur5/visual/*.dae",
            "ManipulaPy_data/ur5/visual/*.stl",
            "ManipulaPy_data/xarm/*.urdf",
            "ManipulaPy_data/xarm/visual/*.dae",
            "ManipulaPy_data/xarm/visual/*.stl",
            
            # Configuration files
            "*.yaml",
            "*.json",
            "*.xml",
            
            # Documentation assets
            "docs/_static/*",
            
            # Benchmark data and scripts
            "Benchmark/*.py",
            "Benchmark/README.md",
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
