========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |travis| |appveyor|
        |
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/werdich-cfr/badge/?style=flat
    :target: https://readthedocs.org/projects/werdich-cfr
    :alt: Documentation Status

.. |travis| image:: https://travis-ci.org/awerdich/werdich-cfr.svg?branch=master
    :alt: Travis-CI Build Status
    :target: https://travis-ci.org/awerdich/werdich-cfr

.. |appveyor| image:: https://ci.appveyor.com/api/projects/status/github/awerdich/werdich-cfr?branch=master&svg=true
    :alt: AppVeyor Build Status
    :target: https://ci.appveyor.com/project/awerdich/werdich-cfr

.. |version| image:: https://img.shields.io/pypi/v/werdich-cfr.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/werdich-cfr

.. |commits-since| image:: https://img.shields.io/github/commits-since/awerdich/werdich-cfr/v0.0.0.svg
    :alt: Commits since latest release
    :target: https://github.com/awerdich/werdich-cfr/compare/v0.0.0...master

.. |wheel| image:: https://img.shields.io/pypi/wheel/werdich-cfr.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/werdich-cfr

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/werdich-cfr.svg
    :alt: Supported versions
    :target: https://pypi.org/project/werdich-cfr

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/werdich-cfr.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/werdich-cfr


.. end-badges

Models to predict CFR from echocardiography.

* Free software: MIT license

Installation
============

::

    pip install werdich-cfr

Documentation
=============


https://werdich-cfr.readthedocs.io/


Development
===========

To run the all tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
