from setuptools import setup, find_packages

setup(
    name='ManipulaPy',
    version='1.0.0',
    author='Mohamed Aboelnar',
    author_email='aboelnasr1997@gmail.com',
    packages=find_packages(),
    description='A package for serial manipulator operations including kinematics, dynamics, and path planning',
    long_description=open('README.md').read(),
    long_description_content_type='markdown',
    install_requires=[
        'numpy',
        'modern_robotics', 
        'urdfpy',
        'pybullet',
        'networkx==2.4'
    ],
)
