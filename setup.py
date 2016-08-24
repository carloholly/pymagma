#
# Copyright (c) 2016, Carlo Holly.
# All rights reserved.
#

#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name="PackageName", ext_modules=[Extension("pymagma",
                                                 ["pycusp.cpp"],
                                                 include_dirs = ["src", boost_inc_dir],
                                                 library_dirs = ["/", boost_lib_dir],
                                                 libraries = ["boost_python"],
                                                 extra_compile_args = ["-g"])])