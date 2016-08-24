.. -*- rst -*-

pymagma
_______

A Python interface to the MAGMA libraries.
This package is developed to provide an python interface to GPU accelerated matrix and vector operations - especially the sparse-iter functions from MAGMA. It provides easy access to iterative sparse solvers on the GPU. The package currently supports MAGMA-2.0.2. It has not been tested for other versions of the MAGMA libraries. Parts of the code are taken from `scikit-cuda <https://github.com/lebedov/scikit-cuda>`_ written by `Lev Givon <http://www.columbia.edu/~lev/>`_.

The package is written and maintained by `Carlo Holly <https://github.com/carloholly>`_ for my research at RWTH Aachen University at the `Chair for Laser Technology <http://www.llt.rwth-aachen.de>`_.

Prerequisites
_____________

The MAGMA-2.0.2 libraries (libmagma.so and libmagma_sparse.so on a Linux system) have to be installed on your system. To download and install the libraries visit `this site <http://icl.cs.utk.edu/magma/software/view.html?id=244>`_.

Installation
____________

To install pymagma use...

Contact
_______

This project is maintained by Carlo Holly. Please contact me via carlo.holly@rwth-aachen.de.

Citing
______

If you make use of pymagma in your research, please cite it. The BibTeX reference is

    @article{Holly_pymagma,
      author        = {Carlo Holly and Lev E. Givon},
      title         = {pymagma 0.0.1 - a {Python} interface to the {MAGMA} libraries},
      month         = August,
      year          = 2016,
      doi           = "",
      url           = "",
    }

License
_______

This software is licensed under the `BSD License <http://www.opensource.org/licenses/bsd-license.php>`_.
See the included `LICENSE`_ file for more information.

.. _LICENSE: LICENSE.rst
