#!/usr/bin/env python3

# -*- coding: utf-8 -*-
#
# setup.py
#
# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from setuptools import find_packages
from setuptools import setup

required_packages = [
    "boto3>=1.26.28,<2.0",
    "numpy>=1.9.0,<2.0",
    "graph-notebook==3.7.0",
    "scikit-learn>=0.24.1",
    "sagemaker>=2.48.1",
]

VERSION = '0.0.2.dev5'
setup(
    name='neptuneml-toolkit',
    version=VERSION,
    description='Open source library for training models with Amazon Neptune ML',
    maintainer='Neptune ML Team',
    maintainer_email='sojiadeshina@gmail.com',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
    ],
    license='Apache 2.0',
    install_requires=required_packages,
)
