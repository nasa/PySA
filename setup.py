"""
Copyright © 2023, United States Government, as represented by the Administrator
of the National Aeronautics and Space Administration. All rights reserved.

The PySA, a powerful tool for solving optimization problems is licensed under
the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
version = '0.1.0'

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Get requirements from requirements.txt
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    install_requires = [x.strip() for x in f.readlines()]

setup(
    name='pysa',
    version=version,
    description='Fast Simulated Annealing Implemented in Native Python',
    long_description=long_description,
    url='https://github.com/s-mandra/pysa',
    author='Salvatore Mandrà',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='simulator simulated annealing',
    packages=find_packages(exclude=['docs', 'tests']),
    install_requires=install_requires,
    project_urls={  # Optional
        'Bug Reports': 'https://github.com/s-mandra/pysa/issues',
        'Source': 'https://github.com/s-mandra/pysa/',
    },
    include_package_data=True,
)
