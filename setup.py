"""Legacy setuptools shim. NOT the release path.

pyproject.toml's [project] table is canonical. pip and `python -m build`
(PEP 517) read it and ignore the metadata below; only a direct
`python setup.py sdist` uses these values. They mirror pyproject.toml's
metadata (version, description, license, dependencies, extras, keywords,
classifiers, URLs); tests/test_packaging_metadata.py fails if they drift.
Package data still follows the legacy MANIFEST.in layout.

Build releases with `python -m build`, via the publish workflow.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ManipulaPy",
    # Version is canonical in pyproject.toml; this duplicate is kept synced
    # for any legacy tooling that still parses setup.py directly. pip and
    # `python -m build` use pyproject.toml's [project] table, not this.
    version="1.4.2",
    author="Mohamed Aboelnasr",
    author_email="aboelnasr1997@gmail.com",
    description='Differentiable, GPU-accelerated robotic manipulator analysis and control — kinematics, dynamics, planning and control on NumPy, CuPy, PyTorch, or JAX',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/boelnasr/ManipulaPy",
    license='AGPL-3.0-or-later',
    
    # FIXED: Properly find all packages including ManipulaPy_data
    packages=find_packages(
        include=["ManipulaPy", "ManipulaPy.*"],
        exclude=["ManipulaPy.ManipulaPy_data._source*"],
    ),
    
    # Legacy setup.py path mirrors pyproject.toml's lightweight core install.
    # Heavy/platform-specific dependencies live in extras below.
    install_requires=[
        'numpy>=2.0,<3.0',
        'scipy>=1.13',
        'matplotlib>=3.9',
        'numba>=0.60,<0.67',  # deliberate: 0.67 admits NumPy 2.5 (CHANGELOG 1.4.1)
        'pillow>=8.0.0',
    ],

    # Optional dependencies for enhanced functionality. Keep ultralytics at
    # 8.4.0+ so the vision extra works with the NumPy 2.x runtime range.
    extras_require={
        'minimal': [
            'numpy>=2.0,<3.0',
            'scipy>=1.13',
            'matplotlib>=3.9',
            'numba>=0.60,<0.67',  # deliberate: 0.67 admits NumPy 2.5 (CHANGELOG 1.4.1)
            'pybullet>=3.2.5,<4.0',
        ],
        'simulation': [
            'pybullet>=3.2.5,<4.0',
        ],
        'urdf': [
            'trimesh>=3.9.14',
        ],
        'vision': [
            'opencv-python>=4.8.1.78',
            'ultralytics>=8.4.0',
            'torch>=2.7.1',
        ],
        'ml': [
            'torch>=2.7.1',
            'scikit-learn>=1.5.0',
        ],
        'cuda': [
            "cupy-cuda12x>=13.0,<14.0; sys_platform != 'darwin'",
        ],
        'pytorch': [
            'torch>=2.7.1',
        ],
        'jax-cpu': [
            "jax[cpu]>=0.6.0; python_version >= '3.10'",
        ],
        'jax-cuda': [
            "jax[cuda12]>=0.6.0; sys_platform == 'linux' and python_version >= '3.10'",
        ],
        'all': [
            'pybullet>=3.2.5,<4.0',
            'trimesh>=3.9.14',
            'opencv-python>=4.8.1.78',
            'ultralytics>=8.4.0',
            'torch>=2.7.1',
            'scikit-learn>=1.5.0',
            "cupy-cuda12x>=13.0,<14.0; sys_platform != 'darwin'",
            "jax[cpu]>=0.6.0; python_version >= '3.10'",
        ],
        'gpu-cuda11': [
            "cupy-cuda11x>=13.0,<14.0; sys_platform != 'darwin'",
        ],
        'gpu-cuda12': [
            "cupy-cuda12x>=13.0,<14.0; sys_platform != 'darwin'",
        ],
        'gpu-rocm': [
            'cupy-rocm-4-3>=10.0.0',
        ],
        'gpu-pycuda': [
            'pycuda>=2021.1',
        ],
        'vision-headless': [
            'opencv-python-headless>=4.8.1.78',
            'ultralytics>=8.4.0',
            'pillow>=8.0.0',
        ],
        'dev': [
            'pytest>=6.0.0',
            'pytest-cov>=3.0.0',
            'pytest-mock>=3.6.0',
            'pytest-xvfb>=2.0.0',
            'pytest-timeout>=2.1.0',
            'pytest-benchmark>=4.0.0',
            'coverage[toml]>=6.0.0',
            'black>=22.0.0',
            'flake8>=4.0.0',
            'isort>=5.10.0',
            'mypy>=0.950',
            'pre-commit>=2.15.0',
            'tox>=4.0.0',
            "tomli>=1.2.0;python_version<'3.11'",
        ],
        'docs': [
            'sphinx>=8.0.0,<9.0.0',
            'pydata-sphinx-theme>=0.18.0,<0.20.0',
            'sphinx-rtd-theme>=3.0.0,<4.0.0',
            'myst-parser>=4.0.0,<5.0.0',
            'sphinx-autodoc-typehints>=3.0.0,<4.0.0',
            'nbsphinx>=0.8.0',
            'jupyter>=1.0.0',
        ],
        'ci': [
            'pytest>=6.0.0',
            'pytest-cov>=3.0.0',
            'pytest-mock>=3.6.0',
            'pytest-xvfb>=2.0.0',
            'pytest-timeout>=2.1.0',
            'coverage[toml]>=6.0.0',
            'opencv-python-headless>=4.8.1.78',
            "tomli>=1.2.0;python_version<'3.11'",
        ],
        'test-minimal': [
            'ManipulaPy[minimal,dev]',
        ],
        'test-full': [
            'ManipulaPy[dev]',
        ],
        'test-ci': [
            'ManipulaPy[ci,vision-headless]',
        ],
    },
    
    # Package classification
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
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
        'Homepage': 'https://github.com/boelnasr/ManipulaPy',
        'Documentation': 'https://manipulapy.readthedocs.io/',
        'Repository': 'https://github.com/boelnasr/ManipulaPy.git',
        'Bug Tracker': 'https://github.com/boelnasr/ManipulaPy/issues',
        'Changelog': 'https://github.com/boelnasr/ManipulaPy/blob/main/CHANGELOG.md',
        'Funding': 'https://github.com/sponsors/boelnasr',
    },
    
    # Keywords for discoverability
    keywords=[
        'robotics',
        'kinematics',
        'inverse-kinematics',
        'dynamics',
        'manipulator',
        'trajectory-planning',
        'motion-planning',
        'simulation',
        'control-systems',
        'differentiable-robotics',
        'differentiable-simulation',
        'autodiff',
        'jax',
        'pytorch',
        'cupy',
        'gpu',
        'cuda',
        'computer-vision',
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
