#!/usr/bin/env python3

# The MIT License (MIT)
# Copyright (c) 2023 by the xcube team and contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import yaml

from setuptools import setup, find_packages

# Same effect as "from xcube import version", but avoids importing xcube:
version = None
with open('xcube/version.py') as fp:
    exec(fp.read())

with open('README.md') as fp:
    description = fp.read()

with open("environment.yml") as fp:
    environment = yaml.safe_load(fp)

excluded_requirements = [
    "flake8",         # qa only
    "moto",           # testing only
    "openssl",        # not on PyPI
    "pytest",         # testing only
    "pytest-cov",     # testing only
    "python",         # included by default
    "python-blosc",   # not on PyPI
    "requests-mock",  # testing only
]

requirements = [r for r in environment["dependencies"]
                if (isinstance(r, str)
                    and r not in excluded_requirements
                    and not any([r.startswith(er + " ")
                                 for er in excluded_requirements]))]

packages = find_packages(exclude=["test", "test.*"])

# noinspection PyTypeChecker
setup(
    name="xcube",
    version=version,
    description=description,
    license='MIT',
    author='xcube Development Team',
    packages=packages,
    package_data={
        'xcube.webapi.meta': [
            'data/openapi.html'
        ],
        'xcube.webapi.viewer': [
            'data/*', 'data/**/*'
        ]
    },
    entry_points={
        'console_scripts': [
            # xcube's CLI
            'xcube = xcube.cli.main:main',
        ],
        'xcube_plugins': [
            # xcube's default extensions
            'xcube = xcube.plugin:init_plugin',
        ],
    },
    install_requires=requirements,
)
