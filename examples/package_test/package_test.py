#
# Copyright (c) 2016, Carlo Holly.
# All rights reserved.
#

#!/usr/bin/env python

"""
Package test script for pymagma.
"""

import sys, os, ctypes
PYMAGMA_PATH = os.getenv('PYMAGMA_PATH', '')
sys.path.append(PYMAGMA_PATH)
import pymagma
from pymagma import cudart
from pymagma import *
from scipy.sparse import *
from scipy import *
import scipy as sp

def test_spmv(queue):
    Magma_DEV = magma_location_t.Magma_DEV
    Magma_CPU = magma_location_t.Magma_CPU
    Magma_CSR = magma_storage_t.Magma_CSR

    nnz = 6
    m = 3

    # Test sparse matrix:
    # matrix([[1, 0, 2],
    #         [0, 0, 3],
    #         [4, 5, 6]]
    #row = array([0,0,1,2,2,2], dtype=int32)
    indptr = array([0,2,3,6], dtype=int32)
    #col = array([0,2,2,0,1,2], dtype=int32)
    indices = array([0,2,2,0,1,2], dtype=int32)
    val = array([1,2,3,4,5,6], dtype=complex128)
    #val = array([1,2,3,4,5,6], dtype=float64)
    x_ar = array([1,1,1], dtype=complex128)
    y_ar = zeros((m,), dtype=complex128)

    alpha = cuda.cuDoubleComplex()
    alpha.x = 1
    alpha.y = 0
    beta = cuda.cuDoubleComplex()
    beta.x = 1
    beta.y = 0
    A   = magma_t_matrix(Magma_CSR)
    x   = magma_z_matrix(Magma_CSR)
    y   = magma_z_matrix(Magma_CSR)
    A_d = magma_t_matrix(Magma_CSR)
    x_d = magma_z_matrix(Magma_CSR)
    y_d = magma_z_matrix(Magma_CSR)

    #magma_zm_5stencil(10, A, queue)
    magma_zcsrset(m, m, indptr, indices, val, A, queue)
    magma_zvset(m, 1, x_ar, x, queue)
    magma_zvset(m, 1, y_ar, y, queue)

    print 'Matrix A info:'
    print 'nnz: ', A.nnz, ', num_rows: ', A.num_rows, ', num_cols: ', A.num_cols
    print 'memory_location: ', A.memory_location == Magma_CPU
    print A.storage_type, magma_storage_t.Magma_CSR

    m_get, n_get, row_get, col_get, val_get = magma_zcsrget(A, queue)
    print row_get
    print col_get
    print val_get
    print m_get
    print n_get

    magma_zmtransfer(A, A_d, Magma_CPU, Magma_DEV, queue)
    magma_zmtransfer(x, x_d, Magma_CPU, Magma_DEV, queue)
    magma_zmtransfer(y, y_d, Magma_CPU, Magma_DEV, queue)

    magma_z_spmv(alpha, A_d, x_d, beta, y_d, queue)

    magma_zmtransfer(y_d, y, Magma_DEV, Magma_CPU, queue)

    print magma_zvget(y, queue)

    magma_zmfree(A_d, queue)
    magma_zmfree(x_d, queue)
    magma_zmfree(y_d, queue)


def test_magma_vector():
    val = array([1,2,3,4,5,6], dtype=complex128)
    vector_d = magma_vector(val)
    print 'iamax: ', vector_d.iamax()
    print 'to_ndarray: ', vector_d.to_ndarray()

    magma.free(vector_d)


def test_magma_matrix(queue):
    # matrix([[1, 0, 2],
    #         [0, 0, 3],
    #         [4, 5, 6]]
    dtype = np.complex128
    m = 3
    print 'calling test_magma_matrix for ', dtype, '...'
    data = array([1+1j,2+1j,3+1j,4+1j,5+1j,6+1j], dtype=dtype)
    row  = array([0,0,1,2,2,2], dtype = np.int32)
    col  = array([0,2,2,0,1,2], dtype = np.int32)
    A_sp = csr_matrix((data,(row,col)), shape=(m,m))
    x_np = array([1,1,1], dtype=dtype)
    y_np = zeros((m,), dtype=dtype)
    A    = magma_matrix(A_sp, queue)
    x    = magma_matrix(x_np, queue)
    y    = magma_matrix(y_np, queue)
    alpha = 1+0j
    beta  = 0j

    magma_spmv(alpha, A, x, beta, y, queue)

    print 'A values: ', A.get_data()
    print 'x values: ', x.get_data()
    print 'y values: ', y.get_data()

    free(A)
    free(x)
    free(y)


def test_solver_magma_matrix(queue):
    Magma_DEV           = magma_location_t.Magma_DEV
    Magma_CPU           = magma_location_t.Magma_CPU
    Magma_CSR           = magma_storage_t.Magma_CSR
    Magma_CUCSR         = magma_storage_t.Magma_CUCSR
    Magma_BICGSTABMERGE = magma_solver_type.Magma_BICGSTABMERGE
    Magma_ILU           = magma_solver_type.Magma_ILU
    Magma_ITERREF       = magma_solver_type.Magma_ITERREF

    dtype = np.complex128
    real  = np.float64
    density = 0.00001
    N = 1000
    print 'calling test_solver_magma_matrix for ', dtype, '...'
    #data = array([1+1j,2+1j,3+1j,4+1j,5+1j,6+1j], dtype=dtype)
    #row  = array([0,0,1,2,2,2], dtype = np.int32)
    #col  = array([0,2,2,0,1,2], dtype = np.int32)
    #A_sp = csr_matrix((data,(row,col)), shape=(m,m))

    print 'calling bicgstab for ', dtype, ' with matrix size ', N ,' x ', N ,'...'
    density_ = density / (2.0 - 1.0/N)
    A_sp = sp.sparse.rand(N, N, density=density_, format='csr', dtype=real) + 1j*sp.sparse.rand(N, N, density=density_, format='csr', dtype=real) + sp.sparse.identity(N)
    A_sp = (A_sp + A_sp.transpose())/2
    #A_csc = sparse.csc_matrix(A_csr)

    A   = magma_matrix(A_sp, queue, storage = Magma_CSR)
    B   = magma_matrix(queue = queue, storage = Magma_CSR, dtype = dtype)
    B_d = magma_matrix(queue = queue, storage = Magma_CSR, dtype = dtype)
    x_d = magma_matrix(queue = queue, storage = Magma_CSR, dtype = dtype)
    b_d = magma_matrix(queue = queue, storage = Magma_CSR, dtype = dtype)

    one  = 1+1j
    zero = 0+0j

    opts = magma_opts_default(dtype = dtype)

    opts.solver_par.solver  = Magma_BICGSTABMERGE
    opts.precond_par.solver = Magma_ILU
    opts.scaling            = magma_scale_t.Magma_UNITROW

    B.magma_t_matrix.blocksize = opts.blocksize
    B.magma_t_matrix.alignment = opts.alignment

    magma_solverinfo_init(opts.solver_par, opts.precond_par, queue)

    #magma_zm_5stencil(laplace_size, A, queue)
    # Create A
    magma_mscale(A, opts.scaling, queue)

    if opts.solver_par.solver != Magma_ITERREF:
        magma_precondsetup(A, b_d, opts.solver_par, opts.precond_par, queue)

    magma_mconvert(A, B, Magma_CSR, opts.output_format, queue)

    print( "\n%% matrix info: %d-by-%d with %d nonzeros\n\n" % (int(A.magma_t_matrix.num_rows), int(A.magma_t_matrix.num_cols), int(A.magma_t_matrix.nnz)))

    print("matrixinfo")
    print("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m   ||   stored nnz\n")
    print("%=================================================================================%\n");
    print("  %8d  %8d      %10d             %4d        %10d\n" % (int(B.magma_t_matrix.num_rows), int(B.magma_t_matrix.num_cols), int(B.magma_t_matrix.true_nnz), int(B.magma_t_matrix.true_nnz/B.magma_t_matrix.num_rows), int(B.magma_t_matrix.nnz)))
    print("%=================================================================================%\n")

    magma_mtransfer(B, B_d, Magma_CPU, Magma_DEV, queue)

    # vectors and initial guess
    magma_vinit(b_d, Magma_DEV, A.magma_t_matrix.num_rows, 1, one, queue)
    # magma_zvinit( &x, Magma_DEV, A.num_cols, 1, one, queue );
    # magma_z_spmv( one, B_d, x, zero, b, queue );                 #  b = A x
    # magma_zmfree(&x, queue );
    magma_vinit(x_d, Magma_DEV, A.magma_t_matrix.num_cols, 1, zero, queue)

    info = magma_solver(B_d, b_d, x_d, opts, queue)

    if info != 0:
        print("%%error: solver returned: %s (%d).\n" % (magma_strerror( info ), int(info)) )

    magma_solverinfo(opts.solver_par, opts.precond_par, queue)

    print("precond_info\n")
    print("%%   setup  runtime\n")
    print("  %.6f  %.6f\n" % (opts.precond_par.setuptime, opts.precond_par.runtime))

    free(B_d)
    free(x_d)
    free(b_d)
    magma_solverinfo_free(opts.solver_par, opts.precond_par, queue)


def test_solver(queue):
    Magma_DEV   = magma_location_t.Magma_DEV
    Magma_CPU   = magma_location_t.Magma_CPU
    Magma_CSR   = magma_storage_t.Magma_CSR
    Magma_CUCSR = magma_storage_t.Magma_CUCSR
    Magma_BICGSTABMERGE = magma_solver_type.Magma_BICGSTABMERGE
    Magma_ILU           = magma_solver_type.Magma_ILU
    Magma_ITERREF       = magma_solver_type.Magma_ITERREF

    dtype = np.complex128
    laplace_size = 100

    A   = magma_t_matrix(Magma_CSR)
    B   = magma_t_matrix(Magma_CSR)
    B_d = magma_t_matrix(Magma_CSR)
    x_d = magma_t_matrix(Magma_CSR)
    b_d = magma_t_matrix(Magma_CSR)

    one = cuda.cuDoubleComplex()
    one.x = 1
    one.y = 0
    zero = cuda.cuDoubleComplex()
    zero.x = 0
    zero.y = 0

    zopts = magma_zopts_default()

    zopts.solver_par.solver = Magma_BICGSTABMERGE
    zopts.precond_par.solver = Magma_ILU
    zopts.scaling = magma_scale_t.Magma_UNITROW

    B.blocksize = zopts.blocksize
    B.alignment = zopts.alignment

    magma_zsolverinfo_init(zopts.solver_par, zopts.precond_par, queue)

    magma_zm_5stencil(laplace_size, A, queue)
    magma_zmscale(A, zopts.scaling, queue)

    if zopts.solver_par.solver != Magma_ITERREF:
        magma_z_precondsetup(A, b_d, zopts.solver_par, zopts.precond_par, queue)

    magma_zmconvert(A, B, Magma_CSR, zopts.output_format, queue)

    print( "\n%% matrix info: %d-by-%d with %d nonzeros\n\n" % (int(A.num_rows), int(A.num_cols), int(A.nnz)))

    print("matrixinfo")
    print("%%   size   (m x n)     ||   nonzeros (nnz)   ||   nnz/m   ||   stored nnz\n")
    print("%%============================================================================%%\n");
    print("  %8d  %8d      %10d             %4d        %10d\n" % (int(B.num_rows), int(B.num_cols), int(B.true_nnz), int(B.true_nnz/B.num_rows), int(B.nnz)))
    print("%%============================================================================%%\n")

    magma_zmtransfer(B, B_d, Magma_CPU, Magma_DEV, queue)

    # vectors and initial guess
    magma_zvinit(b_d, Magma_DEV, A.num_rows, 1, one, queue)
    # magma_zvinit( &x, Magma_DEV, A.num_cols, 1, one, queue );
    # magma_z_spmv( one, B_d, x, zero, b, queue );                 #  b = A x
    # magma_zmfree(&x, queue );
    magma_zvinit(x_d, Magma_DEV, A.num_cols, 1, zero, queue)

    info = magma_z_solver(B_d, b_d, x_d, zopts, queue)

    if info != 0:
        print("%%error: solver returned: %s (%d).\n" % (magma_strerror( info ), int(info)) )

    magma_zsolverinfo(zopts.solver_par, zopts.precond_par, queue)

    print("precond_info\n")
    print("%%   setup  runtime\n")
    print("  %.6f  %.6f\n" % (zopts.precond_par.setuptime, zopts.precond_par.runtime))

    magma_zmfree(B_d, queue)
    magma_zmfree(x_d, queue)
    magma_zmfree(b_d, queue)
    magma_zsolverinfo_free(zopts.solver_par, zopts.precond_par, queue)


def main(argv):
    print "pymagma library test..."

    cudart.cudaDeviceReset()
    mem_free, mem_total = cudart.cudaMemGetInfo()
    print 'GPU free memory: ', float(mem_free)/float(mem_total)*100, ' %'
    print 'cuda driver version', str(cudart.cudaDriverGetVersion())
    print 'device number:', str(cudart.cudaGetDevice())

    print 'magma version', magma_version()
    magma_init()
    # queue is of type magma_queue_t
    queue = magma_queue_t()
    magma_queue_create(0, queue)
    print 'magma_queue_get_device: ', magma_queue_get_device(queue)

    #test_solver(queue)
    #test_magma_matrix(queue)
    test_solver_magma_matrix(queue)
    #test_spmv(queue)

    magma_queue_destroy(queue)
    magma_finalize()

    print "finished."

if __name__ == '__main__':
    main(sys.argv[1:])