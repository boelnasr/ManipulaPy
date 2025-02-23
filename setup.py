from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as fh:
    long_description = fh.read()

setup(
    name='ManipulaPy',
    version='1.0.2',
    author='Mohamed Aboelnar',
    author_email='aboelnasr1997@gmail.com',
    description='A package for robotic serial manipulator operations',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/boelnasr/ManipulaPy',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.2',
        'scipy>=1.5.2',
        'urchin>=0.0.27',
        'pybullet>=3.0.6',
        'pycuda>=2021.1',
        'trimesh>=3.9.14'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    include_package_data=True,
    package_data={
        'ManipulaPy': [
            'ManipulaPy_data/ur5/ur5.urdf',
            'ManipulaPy_data/ur5/visual/*.dae',
            'ManipulaPy_data/xarm/xarm6_robot.urdf',
            'ManipulaPy_data/xarm/visual/*.dae'
        ],
    },
)
