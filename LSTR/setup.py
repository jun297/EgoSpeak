from setuptools import find_packages, setup

setup(
    name='LSTR',
    version='0.1',
    packages=find_packages('src') + ['configs'],
    package_dir={'': 'src', 'configs': 'configs'},
)