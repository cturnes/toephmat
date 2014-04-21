/*
    @(#)File:                /kernel/misc.h
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        20 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Declares miscellaneous routines
*/
#pragma once

#include <cmath>
#include <cstdio>
#include <complex>
#include <climits>
#include <cstring>
#include "mex.h"
#include "lapack.h"
#include "blas.h"

#ifndef _CAST
#define _CAST
#define CCAST(X) reinterpret_cast<double*>(X)
#endif
#ifndef _CMPX
#define _CMPX
#define cmpx std::complex<double>
#endif
#ifndef _EPS
#define _EPS
#define EPS_VAL 2.2204e-16
#endif
#ifndef _NMETA
#define _NMETA
#define NUMMETA 11
#endif

// convert a double array to a complex array
void double2complex(cmpx*,double*,double*,int);

// convert a double array to a complex array
void complex2double(double*,double*,cmpx*,int);

// integer maximum
int imin(int,int);

// integer minimum
int imax(int,int);

// compression of low-rank data - double matrices
void Compress(double**,int,int,double**,int,int,int&);

// compression of low-rank data - complex matrices
void Compress(cmpx**,int,int,cmpx**,int,int,int&);

// transpose in-place - double matrices
void TransposeIP(double**,int,int,int&);

// transpose in-place - double matrices
void TransposeIP(cmpx**,int,int,int&);

// checks whether the data within a hierarchical structure is real or complex
bool IsHMatComplex(const mxArray*);

// reads structure data
void readStructData(const mxArray*, cmpx**, int**, int&, int&);

// reads structure data
void readStructData(const mxArray*, double**, int**, int&, int&);

// writes structure data
mxArray* writeStructData(double*, int*, int, int);

// writes structure data
mxArray* writeStructData(cmpx*, int*, int, int);

// factor input data into left and right vectors
template <class T> void FactorData(T* input, int ldin, const int mIn, const int nIn,
                                   T** left, int &ldl, T** right, int &ldr, int &rNew)
{
    // input validation
    if ((input == NULL) || (left == NULL) || (right == NULL))
    {
        mexErrMsgIdAndTxt("MATLAB:misc:FactorData:badInput",
            "One or more essential array inputs are NULL.");
    }
    if (ldin < mIn)
    {
        mexErrMsgIdAndTxt("MATLAB:misc:FactorData:badInput",
            "Supplied leading dimension is smaller than matrix side length.");
    }
    
    // variables for the SVD calculation
    char jobu = 'S', jobvt = 'S';
    int lwork = -1;
    int rMax = imin(mIn, nIn);
    double *s = (double*)mxMalloc(sizeof(double) * rMax);
    
    // set leading dimensions of input data
    ldl = mIn;
    ldr = nIn;
    
    // complex data
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx workopt, *work;
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * rMax);
        int info;

        // allocate for answer vectors
        (*left) = (T*)mxMalloc(sizeof(T) * ldl * rMax);
        cmpx* temp = (cmpx*)mxMalloc(sizeof(cmpx) * ldr * rMax);
        
        // compute SVD
        zgesvd(&jobu, &jobvt, &ldl, &ldr, CCAST(input), &ldin, s,
               CCAST(*left), &ldl, CCAST(temp), &rMax,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &ldl, &ldr, CCAST(input), &ldin, s,
               CCAST(*left), &ldl, CCAST(temp), &rMax,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // determine actual rank of block
        rNew = 0;
        double tol = s[0] * imin(ldl, ldr) * EPS_VAL;
        cmpx *ptr = reinterpret_cast<cmpx*>(*left);
        for (int j = 0; j < rMax; j++)
        {
            if (s[j] < tol)
                break;
            else
            {
                rNew++;
                for (int i = 0; i < ldl; i++)
                    ptr[i + j * ldl] *= sqrt(s[j]);
            }
        }
            
        // reallocate so that only the compressed versions are kept
        (*left) = (T*)mxRealloc(*left, sizeof(T) * ldl * rNew);
        (*right) = (T*)mxMalloc(sizeof(T) * ldr * rNew);
        ptr = reinterpret_cast<cmpx*>(*right);
        for (int j = 0; j < rNew; j++)
        {
            for (int i = 0; i < ldr; i++)
                ptr[i + j * ldr] = sqrt(s[j]) * conj(temp[j + i*rMax]);
        }
        mxFree(temp);
        mxFree(s);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double workopt, *work;
            int info;
            
            // allocate for answer vectors
            (*left) = (T*)mxMalloc(sizeof(T) * ldl * rMax);
            T* temp = (T*)mxMalloc(sizeof(T) * ldr * rMax);
            
            // compute SVD
            dgesvd(&jobu, &jobvt, &ldl, &ldr, (double*)input, &ldin, s,
                   (double*)(*left), &ldl, (double*)temp, &rMax,
                   &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgesvd(&jobu, &jobvt, &ldl, &ldr, (double*)input, &ldin, s,
                   (double*)(*left), &ldl, (double*)temp, &rMax,
                   work, &lwork, &info);
            mxFree(work);
            
            // determine actual rank of block
            rNew = 0;
            double tol = s[0] * imin(ldl, ldr) * EPS_VAL;
            for (int j = 0; j < rMax; j++)
            {
                if (s[j] < tol)
                    break;
                else
                {
                    rNew++;
                    for (int i = 0; i < ldl; i++)
                        (*(*left + i + j * ldl)) *= sqrt(s[j]);
                }
            }
            
            // reallocate so that only the compressed version is kept
            (*left) = (T*)mxRealloc(*left, sizeof(T) * ldl * rNew);
            
            // allocate right vector
            (*right) = (T*)mxMalloc(sizeof(T) * ldr * rNew);
            for (int j = 0; j < rNew; j++)
            {
                for (int i = 0; i < ldr; i++)
                    (*(*right + i + j * ldr)) = sqrt(s[j]) * temp[j + i*rMax];
            }
            mxFree(temp);
            mxFree(s);
            
        }
        else
            mexWarnMsgIdAndTxt("MATLAB:misc:FactorData:invalidDataType",
                               "Cannot factor because data type is not double or cmpx.");
    }
}

// transpose a matrix in place
template <class T> void TransposeInPlace(T** in, const int nRows, const int nCols, int& ldim)
{
    // input validation
    if (in == NULL)
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeInPlace:badInput",
                          "Supplied data is NULL vector.");
    if (ldim < nRows)
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeInPlace:badInput",
                          "Supplied leading dimension is smaller than matrix side length.");
        
    // branch to specialized cases
    if (sizeof(T) == sizeof(double))
        TransposeIP((double**)in, nRows, nCols, ldim);
    else
    {
        if (sizeof(T) == sizeof(cmpx))
            TransposeIP(reinterpret_cast<cmpx**>(in), nRows, nCols, ldim);
        else
            mexErrMsgIdAndTxt("MATLAB:misc:TransposeInPlace:badDataType",
                          "Cannot transpose because data type is not double or cmpx.");
    }
}