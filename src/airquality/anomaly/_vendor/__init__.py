"""Vendored, trimmed copies of third-party metric code.

Code in this package is copied (and in some cases trimmed) from VUS 0.0.6
("Volume Under the Surface", The DATUM Lab, https://github.com/TheDatumOrg/VUS),
distributed under the Apache-2.0 license. The ``affiliation`` subpackage is in
turn the ``affiliation-metrics`` library by Alexis Huet, as bundled in VUS.

Only the pieces needed by :mod:`genias.metrics` are kept here, so the heavy
``vus`` dependency tree (tensorflow, tsfresh, arch, hurst, tslearn, stumpy,
networkx, cython) is no longer required. See ``NOTICE`` in this directory for
attribution and license details.
"""
