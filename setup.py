#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['pip==19.2.3',
'bump2version==0.5.11',
'wheel==0.33.6',
'watchdog==0.9.0',
'flake8==3.7.8',
'tox==3.14.0',
'coverage==4.5.4',
'Sphinx==1.8.5',
'twine==1.14.0',
'tensorflow==2.5.3',
'Keras==2.2.4',
'nibabel',
'matplotlib',
'opencv-python' ]

setup_requirements = [ ]

test_requirements = [ ]

setup(
    author="Jinwoo Hong",
    author_email='jwhong1125@gmail.com',
    python_requires='>=3.5',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    description="Fetal cortical plate segmentation with multiple predictions using fully convolutional networks",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='fetal_cp_seg',
    name='fetal_cp_seg',
    packages=find_packages(include=['fetal_cp_seg', 'fetal_cp_seg.*']),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/jwhong1125/fetal_cp_seg',
    version='0.1.0',
    zip_safe=False,
)
