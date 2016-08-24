#
# Copyright (c) 2016, Carlo Holly.
# All rights reserved.
#

#!/usr/bin/env python

"""
Benchmark script for pymagma.
"""

import sys, os, time
PYMAGMA_PATH = os.getenv('PYMAGMA_PATH', '')
sys.path.append(PYMAGMA_PATH)
import pymagma
from pymagma import *
import scipy as sp
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spilu
import scipy.io
import numpy.random
from matplotlib import pyplot as plt


def report(xk):
    frame = inspect.currentframe().f_back
    print('residual: ', frame.f_locals['resid'])


def solver_info(info):
    if info == 0:
        print ('successful exit')
    elif info > 0:
        print('convergence to tolerance not achieved, number of iterations')
    elif info < 0:
        print('illegal input or breakdown')


def poisson2d(N,dtype='d',format=None):
    """
        Return a sparse matrix for the 2D Poisson problem
        with standard 5-point finite difference stencil on a
        square N-by-N grid.
    """
    if N == 1:
        diags = np.asarray([[4]],dtype=dtype)
        return sp.sparse.dia_matrix((diags,[0]), shape=(1,1)).asformat(format)

    offsets = np.array([0,-N,N,-1,1])

    diags = np.empty((5,N**2),dtype=dtype)

    diags[0] = 4  # main diagonal
    diags[1:] = -1  # all offdiagonals

    diags[3,N-1::N] = 0  # first lower diagonal
    diags[4,N::N] = 0  # first upper diagonal

    return sp.sparse.dia_matrix((diags,offsets),shape=(N**2,N**2)).asformat(format)


CREATE_TEST_MAT = False

if CREATE_TEST_MAT:
    for N in [10,50,100,150,200,250,300,350,400,450,500,550,600,650]:
        A = poisson2d(N,dtype=np.complex128,format='csr')
        sp.io.mmwrite('poisson2d_' + str(N), A)
    exit()

print "\nStarting pymagma benchmark..."

print 'cuda driver version', str(pycusp.cudart.cudaDriverGetVersion())
print 'device number:', str(pycusp.cudart.cudaGetDevice())

np.set_printoptions(suppress=False)

magma_v = magma_version()
print 'Magma  v%d.%d.%d' % magma_v
cudaDeviceReset()
mem_free, mem_total = cudaMemGetInfo()
print 'GPU free memory: ', float(mem_free)/float(mem_total)*100, ' %'

matrix_list = []
rhs_list =[]
#types_list = [np.float32, np.float64, np.complex128]
for N in [10,50,100,150,200,250,300,350,400,450,500,550,600,650]:
    matrix_list.append('poisson2d_' + str(N) + '.mtx')
    rhs_list.append('')

#,'poisson2d_1000.mtx']#'Chevron1.mtx']#,'3Dspectralwave2_complete.mtx']
#rhs_list    = ['']#,'3Dspectralwave2_b.mtx']
#types_str = ['float32', 'float64', 'complex128']
types_list = [np.complex128]
types_str = ['complex128']

rand = np.random.rand
n_solves = 1
resolutions = [100, 1000, 10000, 100000]
maxiter = 10000
tol = 1e-9
density = 0.00001

MULTIPLY = False
BICGSTAB = False
TEST_MATRICES = True
SCIPY    = True
MAGMA    = True
MAGMA_VERBOSE = True

if MAGMA:
    Magma_DEV = magma_location_t.Magma_DEV
    Magma_CPU = magma_location_t.Magma_CPU
    Magma_CSR = magma_storage_t.Magma_CSR
    Magma_SELLP = magma_storage_t.Magma_SELLP
    queue = magma_queue_t()
    magma_queue_create(0, queue)
    alpha = cuDoubleComplex()
    alpha.x = 1
    alpha.y = 0
    beta = cuDoubleComplex()
    beta.x = 0
    beta.y = 0
    magma_init()


if MULTIPLY:
    i = 0
    for dtype in types_list:
        duration_scipy = []
        duration_magma = []

        for N in resolutions:
            print 'calling multiply for ', dtype, ' with matrix size ', N ,' x ', N ,'...'
            A_csr = sp.sparse.rand(N, N, density=density, format='csr', dtype=dtype)

            if SCIPY:
                print 'Multiplying with SCIPY...'
                b  = np.ones((N,), dtype=dtype)
                start_time = time.time()
                for i in xrange(n_solves):
                    x = A_csr*b
                elapsed_time = time.time() - start_time
                duration_scipy.append(elapsed_time/n_solves)
            else:
                duration_scipy.append(None)

            if MAGMA:
                print 'Multiplying with Magma...'
                start_time = time.time()
                A_d = magma_matrix(A_csr, queue = queue, storage = Magma_CSR)
                x_d = magma_matrix(np.zeros((N,), dtype=dtype), queue = queue, storage = Magma_CSR)
                b_d = magma_matrix(np.ones((N,), dtype=dtype), queue = queue, storage = Magma_CSR)
                for j in xrange(n_solves):
                    magma_spmv(alpha, A_d, b_d, beta, x_d)

                elapsed_time = time.time() - start_time
                duration_magma.append(elapsed_time/n_solves)

                free(A_d)
                free(b_d)
                free(x_d)
            else:
                duration_magma.append(None)


        print 'nodes \t\t duration (magma) / s \t duration (cusp) / s \t duration (scipy) / s'
        for j in xrange(len(resolutions)):
            print resolutions[j]**2, ' \t\t ', duration_magma[j], ' \t ', duration_cusp[j], ' \t', duration_scipy[j]

        w, h = plt.figaspect(1.)
        fig = plt.figure(figsize=(w,h))
        ax = fig.add_subplot( 111 )
        ax.plot(resolutions, duration_magma, linestyle = '-', color = 'k', label='magma')
        ax.plot(resolutions, duration_scipy, linestyle = '--', color = 'k', label='scipy')
        ax.set_xlabel( r"matrix size" )
        ax.set_ylabel( r"$t$ / s" )
        #ax.set_title( r"$n_{ref}=$" + str(n_ref[i]) )
        plt.savefig('benchmark_muliply_' + types_str[i] + '.pdf')
        plt.close()

        i += 1

duration_scipy = []
duration_magma = []

if BICGSTAB:
    i = 0

    for dtype in types_list:
        duration_scipy = []
        duration_magma = []

        for N in resolutions:
            print 'calling bicgstab for ', dtype, ' with matrix size ', N ,' x ', N ,'...'
            density_ = density / (2.0 - 1.0/N)

            if dtype == np.complex128 or dtype == np.float64:
                real = np.float64
            elif dtype == np.complex64 or dtype == np.float32:
                real = np.float32

            A_csr = sp.sparse.rand(N, N, density=density_, format='csr', dtype=real) + 1j*sp.sparse.rand(N, N, density=density_, format='csr', dtype=real) + sp.sparse.identity(N, dtype=dtype)
            A_csr = (A_csr + A_csr.transpose())/2
            A_csc = sparse.csc_matrix(A_csr)

            if SCIPY:
                print 'Solving with SCIPY...'
                b  = np.ones((N,), dtype=dtype)

                P_sp = spilu(A_csc, drop_tol=1e-2)
                M = sp.sparse.linalg.LinearOperator(A_csc.shape, lambda x: P_sp.solve(x))
                #M = None

                start_time = time.time()
                for _ in xrange(n_solves):
                    x0 = np.zeros((N,), dtype=dtype)
                    x, info = sp.sparse.linalg.bicgstab(A_csc,
                                                        b,
                                                        x0=x0,
                                                        M=M,
                                                        tol=tol,
                                                        maxiter=maxiter)
                    print 'convergence information', info

                elapsed_time = time.time() - start_time
                duration_scipy.append(elapsed_time/n_solves)
            else:
                duration_scipy.append(None)

            if MAGMA:
                print 'Solving with Magma...'

                blocksize = 32
                alignment = 1

                A_d = magma_matrix(A_csr, queue = queue, memory_location = Magma_DEV, storage = Magma_SELLP, blocksize = blocksize, alignment = alignment)
                b_d = magma_matrix(queue = queue, memory_location = Magma_DEV, dtype = dtype)
                x_d = magma_matrix(queue = queue, dtype = dtype)

                opts                     = magma_opts_default(dtype = dtype)
                opts.blocksize           = blocksize
                opts.alignment           = alignment
                opts.solver_par.solver   = magma_solver_type.Magma_PBICGSTABMERGE
                opts.solver_par.atol     = tol
                opts.solver_par.rtol     = tol
                opts.solver_par.maxiter  = maxiter
                opts.precond_par.solver  = magma_solver_type.Magma_JACOBI
                opts.precond_par.atol    = tol
                opts.precond_par.rtol    = tol
                opts.precond_par.maxiter = maxiter
                opts.output_format       = Magma_SELLP
                #opts.scaling             = magma_scale_t.Magma_UNITROW
                opts.scaling             = magma_scale_t.Magma_NOSCALE

                #magma_mscale(A_d, opts.scaling, queue)

                magma_solverinfo_init(opts.solver_par, opts.precond_par, queue)
                magma_precondsetup(A_d, b_d, opts.solver_par, opts.precond_par, queue)

                #magma_mconvert(A, A_conv, A.storage, opts.output_format, queue)
                #magma_mtransfer(A_conv, A_d, Magma_CPU, Magma_DEV, queue)

                if MAGMA_VERBOSE:
                    print("precond_info\n")
                    print("%%   setup  runtime\n")
                    print("  %.6f  %.6f\n" % (opts.precond_par.setuptime, opts.precond_par.runtime))

                magma_vinit(b_d, Magma_DEV, N, 1, 1+0j, queue)

                start_time = time.time()
                for _ in xrange(n_solves):
                    #x_d = magma_matrix(np.zeros((N,), dtype=dtype), queue = queue, storage = Magma_CSR)
                    magma_vinit(x_d, Magma_DEV, N, 1, 0+0j, queue)
                    info = magma_solver(A_d, b_d, x_d, opts, queue)
                    if info != 0:
                        print("%%error: solver returned: %s (%d).\n" % (magma_strerror( info ), int(info)) )

                elapsed_time = time.time() - start_time
                duration_magma.append(elapsed_time/n_solves)

                if MAGMA_VERBOSE:
                    magma_solverinfo(opts.solver_par, opts.precond_par, queue)

                free(A_d)
                free(b_d)
                free(x_d)
                magma_solverinfo_free(opts.solver_par, opts.precond_par, queue)
            else:
                duration_magma.append(None)

        print '\n\nnodes \t duration (magma) / s \t duration (scipy) / s'
        for j in xrange(len(resolutions)):
            print resolutions[j]**2, ' \t ', duration_magma[j], ' \t', duration_scipy[j]

        w, h = plt.figaspect(1.)
        fig = plt.figure(figsize=(w,h))
        ax = fig.add_subplot( 111 )
        ax.plot(resolutions, duration_magma, linestyle = '-', color = 'k', label='magma')
        ax.plot(resolutions, duration_scipy, linestyle = '--', color = 'k', label='scipy')
        ax.set_xlabel( r"matrix size" )
        ax.set_ylabel( r"$t$ / s" )
        #ax.set_title( r"$n_{ref}=$" + str(n_ref[i]) )
        plt.savefig('benchmark_bicgstab_' + types_str[i] + '.pdf')
        plt.close()

        i += 1


if TEST_MATRICES:
    i = 0
    duration_scipy = []
    duration_magma = []

    for matrix in matrix_list:
        dtype = np.complex128

        A_csr = sparse.csr_matrix(sp.io.mmread(matrix))
        A_csc = sparse.csc_matrix(A_csr)

        M = A_csr.shape[0]
        N = A_csr.shape[1]

        if rhs_list[i] != '':
            b = np.array(sp.io.mmread(rhs_list[i]).todense(), dtype=dtype)
        else:
            #b = np.array(np.zeros(N,), dtype=dtype)
            #b[N/2] = 1+0j
            b = np.array(np.random.rand(N,1), dtype=dtype) + 1j*np.array(np.random.rand(N,1), dtype=dtype)

        print 'calling bicgstab for ', dtype, ' with matrix size ', A_csr.shape[0] ,' x ', A_csr.shape[0] ,'...'

        if SCIPY:
            print 'Solving with SCIPY...'

            P_sp = spilu(A_csc, drop_tol=1e-2)
            M = sp.sparse.linalg.LinearOperator(A_csc.shape, lambda x: P_sp.solve(x))
            #M = None

            start_time = time.time()
            for _ in xrange(n_solves):
                x0 = np.zeros((N,), dtype=dtype)
                x, info = sp.sparse.linalg.bicgstab(A_csc,
                                                    b,
                                                    x0=x0,
                                                    M=M,
                                                    tol=tol,
                                                    maxiter=maxiter)
                                                    #callback = report)
                solver_info(info)

            elapsed_time = time.time() - start_time
            duration_scipy.append(elapsed_time/n_solves)
        else:
            duration_scipy.append(None)

        if MAGMA:
            print 'Solving with Magma...'

            blocksize = 8
            alignment = 1
            storage = Magma_SELLP

            A_d = magma_matrix(A_csr, queue = queue, memory_location = Magma_DEV, storage = storage, blocksize = blocksize, alignment = alignment)
            b_d = magma_matrix(b, queue = queue, memory_location = Magma_DEV)
            x_d = magma_matrix(queue = queue, dtype = dtype)

            opts                     = magma_opts_default(dtype = dtype)
            opts.blocksize           = blocksize
            opts.alignment           = alignment
            opts.solver_par.solver   = magma_solver_type.Magma_PBICGSTABMERGE
            opts.solver_par.atol     = tol
            opts.solver_par.rtol     = tol
            opts.solver_par.maxiter  = maxiter
            opts.precond_par.solver  = magma_solver_type.Magma_JACOBI
            opts.precond_par.atol    = tol
            opts.precond_par.rtol    = tol
            opts.precond_par.maxiter = maxiter
            #opts.precond_par.levels  = 0
            #opts.precond_par.sweeps  = 0
            #opts.precond_par.restart = 20
            opts.output_format       = storage
            #opts.scaling             = magma_scale_t.Magma_UNITROW
            opts.scaling             = magma_scale_t.Magma_NOSCALE

            #magma_mscale(A_d, opts.scaling, queue)

            magma_solverinfo_init(opts.solver_par, opts.precond_par, queue)
            magma_precondsetup(A_d, b_d, opts.solver_par, opts.precond_par, queue)

            #magma_mconvert(A, A_conv, A.storage, opts.output_format, queue)
            #magma_mtransfer(A_conv, A_d, Magma_CPU, Magma_DEV, queue)

            if MAGMA_VERBOSE:
                print("precond_info\n")
                print("%%   setup  runtime\n")
                print("  %.6f  %.6f\n" % (opts.precond_par.setuptime, opts.precond_par.runtime))

            start_time = time.time()
            for _ in xrange(n_solves):
                #x_d = magma_matrix(np.zeros((N,), dtype=dtype), queue = queue, storage = Magma_CSR)
                magma_vinit(x_d, Magma_DEV, N, 1, 0+0j, queue)
                info = magma_solver(A_d, b_d, x_d, opts, queue)
                if info != 0:
                    print("%%error: solver returned: %s (%d).\n" % (magma_strerror( info ), int(info)) )

            elapsed_time = time.time() - start_time
            duration_magma.append(elapsed_time/n_solves)

            if MAGMA_VERBOSE:
                magma_solverinfo(opts.solver_par, opts.precond_par, queue)

            free(A_d)
            free(b_d)
            free(x_d)
            magma_solverinfo_free(opts.solver_par, opts.precond_par, queue)
        else:
            duration_magma.append(None)

        i += 1

    x_plot = np.arange(len(duration_magma))
    w, h = plt.figaspect(1.)
    fig = plt.figure(figsize=(w,h))
    ax = fig.add_subplot( 111 )
    ax.plot(x_plot, duration_magma, linestyle = '-', color = 'k', label='magma')
    ax.plot(x_plot, duration_scipy, linestyle = '--', color = 'k', label='scipy')
    ax.set_xlabel( r"matrix size" )
    ax.set_ylabel( r"$t$ / s" )
    #ax.set_title( r"$n_{ref}=$" + str(n_ref[i]) )
    plt.savefig('benchmark_bicgstab_test_matrices.pdf')
    plt.close()

    print '\n\nmatrix \t duration (magma) / s \t duration (scipy) / s'
    for j in xrange(len(duration_magma)):
        print matrix_list[j], ' \t ', duration_magma[j], ' \t', duration_scipy[j]


if MAGMA:
    magma_queue_destroy(queue)
    magma_finalize()

print "finished."