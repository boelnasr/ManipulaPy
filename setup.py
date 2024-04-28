#!/usr/bin/env python3


from setuptools import setup, find_packages

setup(
    name='ManipulaPy',
    version='1.0.0',
    author='Mohamed Aboelnar',
    author_email='your-email@example.com',  # Optional
    description='A package for robotic serial manipulator operations',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',  # if your README is in markdown
    url='https://github.com/yourusername/serial_manipulator',  # Optional
    packages=find_packages(),
    install_requires=[
        # Add your package dependencies here
        'numpy',
        'scipy',
        'urchin',
        'pybullet'
        # 'pybullet', # If this is a dependency, it should be listed here
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
