====
spib
====


.. image:: https://img.shields.io/pypi/v/spib.svg
        :target: https://pypi.python.org/pypi/spib

.. image:: https://img.shields.io/travis/wangdedi1997/spib.svg
        :target: https://travis-ci.com/wangdedi1997/spib

.. image:: https://readthedocs.org/projects/spib/badge/?version=latest
        :target: https://spib.readthedocs.io/en/latest/?version=latest
        :alt: Documentation Status



State Predictive Information Bottleneck (SPIB)

* Author: Dedi Wang
* Free software: MIT license
* Documentation: https://spib.readthedocs.io.


What is it?
-----------

SPIB is a deep learning-based framework that learns the reaction coordinates from high dimensional molecular simulation trajectories. Please read and cite this manuscript when using SPIB: https://aip.scitation.org/doi/abs/10.1063/5.0038198. Here is an implementation of SPIB in Pytorch.


Known Issues
------------

* Our implementation now only supports the npy files as the input, and also saves all the results into npy files for further anlyses. Users can refer to the data files in ```scripts/examples```.


Credits
-------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage
