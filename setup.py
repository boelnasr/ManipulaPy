from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="ManipulaPy",
    version="1.1.0",  # Updated version with vision, perception, and simulation modules
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
        "pycuda>=2021.1",
        "trimesh>=4.0,<4.2",
        "opencv-python>=4.5,<5.0",
        "scikit-learn>=1.3,<1.6",
        "matplotlib"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    include_package_data=True,
    package_data={
        "ManipulaPy": [
            "ManipulaPy_data/ur5/ur5.urdf",
            "ManipulaPy_data/ur5/visual/*.dae",
            "ManipulaPy_data/xarm/xarm6_robot.urdf",
            "ManipulaPy_data/xarm/visual/*.dae"
        ],
    },
    project_urls={
        "Documentation": "https://github.com/boelnasr/ManipulaPy",
        "Source": "https://github.com/boelnasr/ManipulaPy",
        "Tracker": "https://github.com/boelnasr/ManipulaPy/issues"
    },
    keywords=["robotics", "kinematics", "dynamics", "trajectory", "control", "simulation", "pybullet", "vision"]
)