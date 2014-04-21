/*
    @(#)File:                /kernel/lrmat.h
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        20 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Declares routines for low-rank matrices
*/
#pragma once

#include <cmath>
#include <cstdio>
#include <complex>
#include <cstring>
#include "mex.h"
#include "lapack.h"
#include "blas.h"
#include "misc.h"

#ifndef _CAST
#define _CAST
#define CCAST(X) reinterpret_cast<double*>(X)
#endif
#ifndef _EPS
#define _EPS
#define EPS_VAL 2.2204e-16
#endif
#ifndef _CMPX
#define _CMPX
#define cmpx std::complex<double>
#endif
#ifndef _RANKDEFS
#define _RANKDEFS
#endif
#ifndef _BLOCKSIZE
#define BLOCKSIZE 64
#endif

using namespace std;

/*  low-rank matrix class */
template <class T> class LowRankMat {
    
    public:
    
        // default initialization
        LowRankMat()
        {
            this->leftVectors = NULL;
            this->rightVectors = NULL;
            this->ldLeft = 0;
            this->ldRight = 0;
            this->rank = 0;
            this->m = 0;
            this->n = 0;
        }
        
        // destructor
        ~LowRankMat()
        {
            if (this->leftVectors != NULL)
            {
                mxFree(this->leftVectors);
                this->leftVectors = NULL;
            }
            if (this->rightVectors != NULL)
            {
                mxFree(this->rightVectors);
                this->rightVectors = NULL;
            }
            this->ldLeft = 0;
            this->ldRight = 0;
            this->rank = 0;
            this->m = 0;
            this->n = 0;
        }
        
        // size initialization
        LowRankMat(const int inLdLeft, const int inM,
                   const int inLdRight, const int inN,
                   const int inRank)
        {
            // default init
            this->leftVectors = NULL;
            this->rightVectors = NULL;
            this->ldLeft = 0;
            this->ldRight = 0;
            this->rank = 0;
            this->m = 0;
            this->n = 0;
            
            // validate inputs
            if ((inM < 1) || (inN < 1))
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                    "One or more supplied input data lengths are less than 1.");
                
            if ((inLdLeft < inM) || (inLdRight < inN))
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                    "One or more supplied leading dimensions are less than matrix dimensions.");
                
            if (inRank < 1)
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                                  "Supplied input data rank is negative.");
            
            // assign if inputs are valid
            this->ldLeft = inLdLeft;
            this->ldRight = inLdRight;
            this->rank = inRank;
            this->m = inM;
            this->n = inN;
            
            // allocate for vectors
            this->leftVectors = (T*)mxMalloc(sizeof(T) * inLdLeft * inRank);
            memset(this->leftVectors, 0, sizeof(T) * inLdLeft * inRank);
            this->rightVectors = (T*)mxMalloc(sizeof(T) * inLdRight * inRank);
            memset(this->rightVectors, 0, sizeof(T) * inLdRight * inRank);
        }
        
        // copy constructor
        LowRankMat(LowRankMat const& rhs)
        {
            // copy init
            this->ldLeft = rhs.ldLeft;
            this->ldRight = rhs.ldRight;
            this->rank = rhs.rank;
            this->m = rhs.m;
            this->n = rhs.n;
            if (rhs.leftVectors != NULL)
            {
                this->leftVectors = (T*)mxMalloc(sizeof(T) * this->ldLeft * this->rank);
                memcpy(this->leftVectors, rhs.leftVectors, sizeof(T) * rhs.ldLeft * rhs.rank);
            }
            else
                this->leftVectors = NULL;
            if (rhs.rightVectors != NULL)
            {
                this->rightVectors = (T*)mxMalloc(sizeof(T) * this->ldRight * this->rank);
                memcpy(this->rightVectors, rhs.rightVectors, sizeof(T) * rhs.ldRight * rhs.rank);
            }
            else
                this->rightVectors = NULL;
        }
        
        // constructor from individual generators
        LowRankMat(T* left, const int ldl, const int inM,
                   T* right, const int ldr, const int inN, const int rIn)
        {
            // input validation
            if ((left == NULL) || (right == NULL))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                                  "One or more supplied input data arrays are NULL.");
            }
            if ((ldl < inM) || (ldr < inN))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                    "One or more supplied leading dimensions are less than matrix dimensions.");
            }
            if (rIn < 1)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                    "Input rank is an invalid value.");
            }
            
            // allocate:
            this->leftVectors = (T*)mxMalloc(sizeof(T) * inM * rIn);
            memset(this->leftVectors, 0, sizeof(T) * inM * rIn);
            this->rightVectors = (T*)mxMalloc(sizeof(T) * inN * rIn);
            memset(this->rightVectors, 0, sizeof(T) * inN * rIn);
            
            // copy data
            for (int i = 0; i < rIn; i++)
            {
                memcpy(this->leftVectors + i*inM, left + i*ldl, sizeof(T) * inM);
                memcpy(this->rightVectors + i*inN, right + i*ldr, sizeof(T) * inN);
            }
            this->m = inM;
            this->n = inN;
            this->ldLeft = inM;
            this->ldRight = inN;
            this->rank = rIn;
        }
        
        // constructor from a full matrix
        LowRankMat(T* const source, const int ldIn, const int mIn, const int nIn)
        {
            // input validation
            if (source == NULL)
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                    "Input data array is NULL.");
            if ((mIn < 1) || (nIn < 1))
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                    "Specified dimensions of input data array are invalid.");
            if (ldIn < mIn)
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:badInput",
                    "Specified leading dimension of data out of range.");
                
            // set defaults
            this->rank = 0;
            this->leftVectors = NULL;
            this->rightVectors = NULL;
            
            // set dimensions and storage sizes
            this->m = mIn;
            this->n = nIn;
            this->ldLeft = this->m;
            this->ldRight = this->n;
            
            // factor and compress the data
            T* sourceCopy = (T*)mxMalloc(sizeof(T) * mIn * nIn);
            for (int i = 0; i < nIn; i++)
                memcpy(sourceCopy + i * mIn, source + i * ldIn, sizeof(T) * mIn);
            FactorData(sourceCopy, mIn, mIn, nIn, &(this->leftVectors), this->ldLeft,
                       &(this->rightVectors), this->ldRight, this->rank);
            mxFree(sourceCopy);
        }
        
        // returns matrix dimensions
        void Dims(int *a) const
        {
            if (a == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Dims:badInput",
                                  "Supplied vector to put dimensions into is NULL.");
            }
            else
            {
                a[0] = this->m;
                a[1] = this->n;
            }
            return;
        }
        
        // copies the decomposition data into a specified set of arrays
        void CopyData(T* left, const int ldl, T* right, const int ldr) const
        {
            // input validation
            if ((left == NULL) || (right == NULL))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                                  "One or more supplied input data arrays are NULL.");
            }
            if ((ldl < this->m) || (ldr < this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                    "One or more supplied leading dimensions are less than matrix dimensions.");
            }
            
            // data copy
            if (this->leftVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(left + i*ldl, this->leftVectors + i*this->ldLeft,
                           sizeof(T) * this->m);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:emptyInternalData",
                                   "Cannot copy left vectors because internal array is NULL.");
            }
            
            if (this->rightVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(right + i*ldr, this->rightVectors + i*this->ldRight,
                           sizeof(T) * this->n);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:emptyInternalData",
                                   "Cannot copy right vectors because internal array is NULL.");
            }
        }
        
        // copies a submatrix of the decomposition data into a specified set of arrays
        void CopyData(T* left, const int ldl, T* right, const int ldr, const int rI, const int cI,
                      const int rLen, const int cLen) const
        {
            // input validation
            if ((left == NULL) || (right == NULL))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                                  "One or more supplied input data arrays are NULL.");
            }
            // check input column and row indices
            if ((rLen < 1) || (cLen < 1))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                                  "One or more supplied copy lengths are invalid.");
            }
            if ((rI < 0) || (rI >= this->m))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                                  "Supplied row index is out of range.");
            }
            if ((cI < 0) || (cI >= this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                                  "Supplied column index is out of range.");
            }
            if (((rI + rLen - 1) > m) || ((cI + cLen - 1) > n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                                  "Supplied data length is out of range.");
            }
            if ((ldl < rLen) || (ldr < cLen))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:badInput",
                    "One or more supplied leading dimensions are less than matrix dimensions.");
            }
            
            // data copy
            if (this->leftVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(left + i*ldl, this->leftVectors + i*this->ldLeft + rI,
                           sizeof(T) * rLen);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:emptyInternalData",
                                   "Cannot copy left vectors because internal array is NULL.");
            }
            
            if (this->rightVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(right + i*ldr, this->rightVectors + i*this->ldRight + cI,
                           sizeof(T) * cLen);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyData:emptyInternalData",
                                   "Cannot copy right vectors because internal array is NULL.");
            }
        }
        
        // copies a submatrix of the left vectors into a specified array
        void CopyLeftData(T* left, const int ldl, const int rI, const int rLen) const
        {
            // input validation
            if (left == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyLeftData:badInput",
                                  "Supplied input data array is NULL.");
            }
            // check input column and row indices
            if (rLen < 1)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyLeftData:badInput",
                                  "Supplied copy length is invalid.");
            }
            if ((rI < 0) || (rI >= this->m))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyLeftData:badInput",
                                  "Supplied row index is out of range.");
            }
            if ((rI + rLen - 1) > this->m)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyLeftData:badInput",
                                  "Supplied data length is out of range.");
            }
            if (ldl < rLen)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyLeftData:badInput",
                    "Supplied leading dimension is less than copy length.");
            }
            
            // data copy
            if (this->leftVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(left + i*ldl, this->leftVectors + i*this->ldLeft + rI,
                           sizeof(T) * rLen);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyLeftData:emptyInternalData",
                                   "Cannot copy left vectors because internal array is NULL.");
            }
        }
        
        // copies a submatrix of the right vectors into a specified array
        void CopyRightData(T* right, const int ldr, const int cI, const int cLen) const
        {
            // input validation
            if (right == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied input data array is NULL.");
            }
            // check input column and row indices
            if (cLen < 1)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied copy length is invalid.");
            }
            if ((cI < 0) || (cI >= this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied column index is out of range.");
            }
            if ((cI + cLen - 1) > this->n)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied data length is out of range.");
            }
            if (ldr < cLen)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                    "Supplied leading dimension is less than copy length.");
            }
            
            // data copy
            if (this->rightVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(right + i*ldr, this->rightVectors + i*this->ldRight + cI,
                           sizeof(T) * cLen);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:emptyInternalData",
                                   "Cannot copy left vectors because internal array is NULL.");
            }
        }
        
        // copies a submatrix of the right vectors into a specified array
        void CopyRightDataTranspose(T* right, const int ldr, const int cI, const int cLen) const
        {
            // input validation
            if (right == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied input data array is NULL.");
            }
            // check input column and row indices
            if (cLen < 1)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied copy length is invalid.");
            }
            if ((cI < 0) || (cI >= this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied column index is out of range.");
            }
            if ((cI + cLen - 1) > this->n)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                                  "Supplied data length is out of range.");
            }
            if (ldr < this->rank)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:badInput",
                    "Supplied leading dimension is less than the matrix rank.");
            }
            
            // data copy
            if (this->rightVectors != NULL)
            {
                if (sizeof(T) == sizeof(cmpx))
                {
                    cmpx *rightCmpx = reinterpret_cast<cmpx*>(right);
                    for (int j = 0; j < this->rank; j++)
                    {
                        for (int i = 0; i < cLen; i++)
                            rightCmpx[j + i*ldr] = conj(cmpx(this->rightVectors[i + cI + j*this->ldRight]));
                    }
                }
                else
                {
                    for (int j = 0; j < this->rank; j++)
                    {
                        for (int i = 0; i < cLen; i++)
                            right[j + i*ldr] = this->rightVectors[i + cI + j*this->ldRight];
                    }
                }
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CopyRightData:emptyInternalData",
                                   "Cannot copy left vectors because internal array is NULL.");
            }
        }
        
        // sets all of the vectors for the left and right
        void SetVectors(T* left, const int ldl, T* right, const int ldr)
        {
            // input validation
            if ((left == NULL) || (right == NULL))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:badInput",
                                  "One or more supplied input data arrays are NULL.");
            }
            if ((ldLeft < this->m) || (ldRight < this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:badInput",
                    "One or more supplied leading dimensions are less than matrix dimensions.");
            }
            
            // data copy
            if (this->leftVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(this->leftVectors + i*this->ldLeft, left + i*ldl, 
                           sizeof(T) * this->m);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:emptyInternalData",
                                   "Cannot copy left vectors because internal array is NULL.");
            }
            
            if (this->rightVectors != NULL)
            {
                for (int i = 0; i < this->rank; i++)
                    memcpy(this->rightVectors + i*this->ldRight, right + i*ldr, 
                           sizeof(T) * this->n);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:emptyInternalData",
                                   "Cannot copy right vectors because internal array is NULL.");
            }
        }
        
        // sets a subset of vectors for the left and right
        void SetVectors(T* left, const int ldl, T* right, const int ldr, int start, int num)
        {
            // input validation
            if ((left == NULL) || (right == NULL))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:badInput",
                                  "One or more supplied input data arrays are NULL.");
            }
            if ((ldLeft < this->m) || (ldRight < this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:badInput",
                    "One or more supplied leading dimensions are less than matrix dimensions.");
            }
            if ((start < 0) || (start >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:badInput",
                    "Specified starting index is out of range.");
            }
            if ((num < 0) || ((start + num - 1) >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:badInput",
                    "Specified number of columns to assign is invalid.");
            }
            
            // data copy
            if (this->leftVectors != NULL)
            {
                for (int i = start; i < (start + num); i++)
                    memcpy(this->leftVectors + i*this->ldLeft, left + (i - start)*ldl, 
                           sizeof(T) * this->m);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:emptyInternalData",
                                   "Cannot copy left vectors because internal array is NULL.");
            }
            
            if (this->rightVectors != NULL)
            {
                for (int i = start; i < (start + num); i++)
                    memcpy(this->rightVectors + i*this->ldRight, right + (i - start)*ldr, 
                           sizeof(T) * this->n);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetVectors:emptyInternalData",
                                   "Cannot copy right vectors because internal array is NULL.");
            }
        }
        
        // negates all vectors
        void Negate()
        {
            this->Negate(0, this->rank);
        }
        
        // sets a subset of vectors for the left and right
        void Negate(int start, int num)
        {
            if ((start < 0) || (start >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Negate:badInput",
                    "Specified starting index is out of range.");
            }
            if ((num < 0) || ((start + num - 1) >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Negate:badInput",
                    "Specified number of columns to assign is invalid.");
            }
            
            // make data negative
            if (this->leftVectors != NULL)
            {
                for (int j = start; j < (start + num); j++)
                {
                    for (int i = 0; i < this->m; i++)
                        this->leftVectors[i + j*this->ldLeft] = -this->leftVectors[i + j*this->ldLeft];
                }
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Negate:emptyInternalData",
                                   "Cannot negate left vectors because internal array is NULL.");
            }
        }
        
        // gets an element from the left vectors
        T GetLeftElement(const int row, const int col) const
        {
            if ((row < 0) || (row >= this->m))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:GetLeftElement:badIndex",
                    "Supplied row index is out of range.");
            }
            if ((col < 0) || (col >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:GetLeftElement:badIndex",
                    "Supplied column index is out of range.");
            }
            if (this->leftVectors == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:GetLeftElement:emptyInternalData",
                    "Cannot retrieve left vector element because internal array is NULL.");
            }
            
            return this->leftVectors[row + col*this->ldLeft];
        }
        
        // sets an element in the left vectors
        void SetLeftElement(T value, const int row, const int col)
        {
            if ((row < 0) || (row >= this->m))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetLeftElement:badIndex",
                    "Supplied row index is out of range.");
            }
            if ((col < 0) || (col >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetLeftElement:badIndex",
                    "Supplied column index is out of range.");
            }
            if (this->leftVectors == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetLeftElement:emptyInternalData",
                    "Cannot set left vector element because internal array is NULL.");
            }
            
            this->leftVectors[row + col*this->ldLeft] = value;
        }
        
        // gets an element from the right vectors
        T GetRightElement(const int row, const int col) const
        {
            if ((row < 0) || (row >= this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:GetRightElement:badIndex",
                    "Supplied row index is out of range.");
            }
            if ((col < 0) || (col >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:GetRightElement:badIndex",
                    "Supplied column index is out of range.");
            }
            if (this->rightVectors == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:GetRightElement:emptyInternalData",
                    "Cannot retrieve right vector element because internal array is NULL.");
            }
            
            return this->rightVectors[row + col*this->ldRight];
        }
        
        // sets an element in the right vectors
        void SetRightElement(T value, const int row, const int col)
        {
            if ((row < 0) || (row >= this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetRightElement:badIndex",
                    "Supplied row index is out of range.");
            }
            if ((col < 0) || (col >= this->rank))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetRightElement:badIndex",
                    "Supplied column index is out of range.");
            }
            if (this->rightVectors == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SetRightElement:emptyInternalData",
                    "Cannot set right vector element because internal array is NULL.");
            }
            
            this->rightVectors[row + col*this->ldRight] = value;
        }
        
        // returns matrix rank
        int Rank() const { return this->rank; }
        
        // returns a low-rank submatrix
        LowRankMat<T>* Submat(const int rI, const int nRows, const int cI, const int nCols) const
        {
            // input validation
            if (nRows < 1)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Submat:badInput",
                    "Number of rows in desired submatrix must be positive.");
            }
            if (nCols < 1)
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Submat:badInput",
                    "Number of columns in desired submatrix must be positive.");
            }
            if ((rI < 0) || ((rI + nRows - 1) >= this->m) || (cI < 0) || ((cI + nCols - 1) >= this->n))
            {
                mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Submat:badInput",
                    "Specified submatrix parameters are invalid or inconsistent.");
            }
            
            LowRankMat<T>* lhs = new LowRankMat<T>(nRows, nRows, nCols, nCols, this->rank);
            for (int i = 0; i < this->rank; i++)
            {
                memcpy(lhs->leftVectors + i*nRows, this->leftVectors + rI + i*this->ldLeft, sizeof(T) * nRows);
                memcpy(lhs->rightVectors + i*nCols, this->rightVectors + cI + i*this->ldRight, sizeof(T) * nCols);
            }
            return lhs;
        }
        
        // assignment operator
        LowRankMat<T>& operator=(const LowRankMat<T> rhs)
        {
            rhs.Swap(*this);
            return *this;
        }
        
        /************* operator definitions *************/
        
        // add and assign
        LowRankMat<T>& operator += (const LowRankMat<T>);
        
        // subtract and assign
        LowRankMat<T>& operator -= (const LowRankMat<T>);
        
        // add two low-rank matrices
        LowRankMat<T> operator + (const LowRankMat<T>) const ;
        
        // subtract two low-rank matrices
        LowRankMat<T> operator - (const LowRankMat<T>) const ;
        
        // multiply and assign
        LowRankMat<T>& operator *= (const LowRankMat<T>);
        
        // multiply two low-rank matrices
        LowRankMat<T> operator * (const LowRankMat<T>) const ;
        
        /************* specialized addition *************/
        
        // add and assign a submatrix of another low-rank matrix
        void AddAssignPart(LowRankMat<T>*, int, int);
        
        // subtract and assign a submatrix of another low-rank matrix
        void SubAssignPart(LowRankMat<T>*, int, int);
        
        // add a low-rank matrix with a submatrix of another low-rank matrix
        LowRankMat<T> AddPart(LowRankMat<T>*, int, int) const ;
        
        // subtract a low-rank matrix with a submatrix of another low-rank matrix
        LowRankMat<T> SubPart(LowRankMat<T>*, int, int) const ;
        
        /************* specialized multiplication *************/
        
        // right-multiply and assign by a submatrix of another low-rank matrix
        void RMultAssignPart(LowRankMat<T>*,int,int,int);
        
        // right-multiply and assign by a dense matrix
        void RMultAssignPart(T*,int,int);
        
        // left-multiply and assign by a submatrix of another low-rank matrix
        void LMultAssignPart(LowRankMat<T>*,int,int,int);
        
        // left-multiply and assign by a dense matrix
        void LMultAssignPart(T*,int,int);
        
        // right-multiply by a submatrix of another low-rank matrix
        LowRankMat<T>* RMultPart(LowRankMat<T>*,int,int,int) const;
        
        // right-multiply by a dense matrix
        LowRankMat<T>* RMultPart(T*,int,int) const;
        
        // left-multiply by a submatrix of another low-rank matrix
        LowRankMat<T>* LMultPart(LowRankMat<T>*,int,int,int) const;
        
        // left-multiply by a dense matrix
        LowRankMat<T>* LMultPart(T*,int,int) const;
        
        // right-multiply by a dense matrix, returning a dense matrix
        T* RMultDense(T*,int,int,int&) const ;
        
        // left-multiply by a dense matrix, returning a dense matrix
        T* LMultDense(T*,int,int,int&) const ;
        
        /************* other *************/
        
        // make a dense version
        void MakeDense(T**,int&,int,int);
        
        // make a dense version
        void MakeDense(T**,int&,int,int,int,int);
        
        // compress generators
        void CompressVectors()
        {
            if (sizeof(T) == sizeof(cmpx))
            {
                Compress(reinterpret_cast<cmpx**>(&(this->leftVectors)), this->ldLeft, this->m,
                         reinterpret_cast<cmpx**>(&(this->rightVectors)), this->ldRight, this->n,
                         this->rank);
            }
            else
            {
                if (sizeof(T) == sizeof(double))
                {
                    Compress(reinterpret_cast<double**>(&(this->leftVectors)), this->ldLeft, this->m,
                         reinterpret_cast<double**>(&(this->rightVectors)), this->ldRight, this->n,
                         this->rank);
                }
                else
                {
                    mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:CompressVectors:invalidDataType",
                                   "Cannot compress representation because data type is not double or cmpx.");
                }
            }
            return;
        }
    
        // print out information about this matrix
        void About() const
        {
            mexPrintf("--------------------------------------\n");
            mexPrintf("\t Low-rank matrix, %d x %d \t\n", this->m, this->n);
            mexPrintf("\t \t Rank: %d\n", this->rank);
            mexPrintf("\t \t Leading dims: %d and %d\n", this->ldLeft, this->ldRight);
            mexPrintf("--------------------------------------\n");
        }
        
        // returns storage information
        int Storage() const { return ((this->m + this->n) * this->rank); }
    
    private:
        
        // low-rank decomposition data
        T *leftVectors, *rightVectors;
        
        // leading dimension of data vector
        int ldLeft, ldRight;
        
        // rank
        int rank;
        
        // matrix size
        int m, n;
        
        // swap data elements
        void Swap(LowRankMat& s)
        {
            std::swap(this->leftVectors, s.leftVectors);
            std::swap(this->rightVectors, s.rightVectors);
            std::swap(this->ldLeft, s.ldLeft);
            std::swap(this->ldRight, s.ldRight);
            std::swap(this->rank, s.rank);
            std::swap(this->m, s.m);
            std::swap(this->n, s.n);
        }
        
};

// compare the dimensions of two hierarchical matrices
template <class T> bool DimMatch(LowRankMat<T>* rhs1, LowRankMat<T>* rhs2)
{
    // input validation
    if ((rhs1 == NULL) || (rhs2 == NULL))
        mexErrMsgIdAndTxt("MATLAB:lrmat:DimMatch:invalidInput",
                          "One or more input structures are NULL.");
    int dimRhs[4];
    rhs1->Dims(dimRhs);
    rhs2->Dims(dimRhs + 2);
    
    bool success = false;
    if ((dimRhs[0] == dimRhs[2]) && (dimRhs[1] == dimRhs[3]))
        success = true;
    return success;
}

/************* operator definitions *************/

// addition-assignment of low-rank matrices
template <class T> LowRankMat<T>& LowRankMat<T>::operator += (const LowRankMat<T> rhs)
{
    // get matrix dimensions
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:PlusEqual:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // get augmented rank
    int r1 = rhs.Rank();
    
    // reallocate to include space for new data
    int newNumel = this->ldLeft * (this->rank + r1);
    this->leftVectors = (T*)mxRealloc(this->leftVectors, sizeof(T) * newNumel);
    newNumel = this->ldRight * (this->rank + r1);
    this->rightVectors = (T*)mxRealloc(this->rightVectors, sizeof(T) * newNumel);
    
    // add in new data to each side
    rhs.CopyData(this->leftVectors + this->rank * this->ldLeft, this->ldLeft,
                 this->rightVectors + this->rank * this->ldRight, this->ldRight);
    this->rank += r1;
    
    // compress
    this->CompressVectors();
    
    // return
    return (*this);
}

// subtraction-assignment of low-rank matrices
template <class T> LowRankMat<T>& LowRankMat<T>::operator -= (const LowRankMat<T> rhs)
{
    // get matrix dimensions
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MinusEqual:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // get augmented rank
    int r1 = rhs.Rank();
    
    // reallocate to include space for new data
    int newNumel = this->ldLeft * (this->rank + r1);
    this->leftVectors = (T*)mxRealloc(this->leftVectors, sizeof(T) * newNumel);
    newNumel = this->ldRight * (this->rank + r1);
    this->rightVectors = (T*)mxRealloc(this->rightVectors, sizeof(T) * newNumel);
    
    // add in new data to each side
    rhs.CopyData(this->leftVectors + this->rank * this->ldLeft, this->ldLeft,
                 this->rightVectors + this->rank * this->ldRight, this->ldRight);
    int oRank = this->rank;
    this->rank += r1;
    this->Negate(oRank, r1);
    
    // compress
    this->CompressVectors();
    
    // return
    return (*this);
}

// add two low-rank matrices to form a new instance
template <class T> LowRankMat<T> LowRankMat<T>::operator + (const LowRankMat<T> rhs) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Plus:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // get new rank
    int rnew = rhs.Rank() + this->rank;
    
    // create new instance
    LowRankMat<T> lhs(dimRhs[0], dimRhs[0], dimRhs[1], dimRhs[1], rnew);
    
    // copy in vectors
    lhs.SetVectors(this->leftVectors, this->ldLeft, this->rightVectors, this->ldRight, 0, this->rank);
    lhs.SetVectors(rhs.leftVectors, rhs.ldLeft, rhs.rightVectors, rhs.ldRight, this->rank, rhs.rank);
    
    // compress representation
    lhs.CompressVectors();
    
    return lhs;
}

// subtract two low-rank matrices to form a new instance
template <class T> LowRankMat<T> LowRankMat<T>::operator - (const LowRankMat<T> rhs) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Minus:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // get new rank
    int rnew = rhs.Rank() + this->rank;
    
    // create new instance
    LowRankMat<T> lhs(dimRhs[0], dimRhs[0], dimRhs[1], dimRhs[1], rnew);
    
    // copy in vectors
    lhs.SetVectors(this->leftVectors, this->ldLeft, this->rightVectors, this->ldRight, 0, this->rank);
    lhs.SetVectors(rhs.leftVectors, rhs.ldLeft, rhs.rightVectors, rhs.ldRight, this->rank, rhs.rank);
    lhs.Negate(this->rank, rhs.rank);
    
    // compress representation
    lhs.CompressVectors();
    
    return lhs;
}

// multiplication-assignment of low-rank matrices
template <class T> LowRankMat<T>& LowRankMat<T>::operator *= (const LowRankMat<T> rhs)
{
    // get matrix dimensions
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if (this->n != dimRhs[0])
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:TimesEqual:dimMismatch",
                          "Inner matrix dimensions are not equal.");
        
    // perform matrix multiplication and svd
    int rLeft = this->rank, rRight = rhs.rank;
    int rNew = imin(rLeft, rRight);
    T* innerMat = (T*)mxMalloc(sizeof(T) * rLeft * rRight * 2);
    char transA = 'C', transB = 'N';
    char jobu = 'O', jobvt = 'S';
    double *s = (double*)mxMalloc(sizeof(double) * rNew);
    int lwork = -1;
    
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0, workopt, *work;
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * rNew);
        int info;
        
        // matrix multiply
        int ldr = this->ldRight, ldl = rhs.ldLeft;
        zgemm(&transA, &transB, &rLeft, &rRight, dimRhs, CCAST(&alpha), CCAST(this->rightVectors),
              &ldr, CCAST(rhs.leftVectors), &ldl, CCAST(&beta),
              CCAST(innerMat), &rLeft);
        
        // compute SVD
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // populate vectors of new instance
        transA = 'N';
        transB = 'N';
        ldl = this->ldLeft;
        ldr = this->m;
        T* newLeft = (T*)mxMalloc(sizeof(T) * this->m * rNew);
        zgemm(&transA, &transB, &ldr, &rNew, &rLeft, CCAST(&alpha),
              CCAST(this->leftVectors), &ldl, CCAST(innerMat), &rLeft,
              CCAST(&beta), CCAST(newLeft), &ldr);
        // swap in new array
        mxFree(this->leftVectors);
        this->leftVectors = newLeft;
        
        transB = 'C';
        ldl = rhs.ldRight;
        ldr = rhs.n;
        T* newRight = (T*)mxMalloc(sizeof(T) * rhs.n * rNew);
        zgemm(&transA, &transB, &ldr, &rNew, &rRight, CCAST(&alpha),
              CCAST(rhs.rightVectors), &ldl, CCAST(innerMat + rLeft*rRight), &rLeft,
              CCAST(&beta), CCAST(newRight), &ldr);
        // swap in new array
        mxFree(this->rightVectors);
        this->rightVectors = newRight;
        
        // update storage info
        this->n = rhs.n;
        this->ldLeft = this->m;
        this->ldRight = this->n;
        this->rank = rNew;
        
        // modify new vectors
        for (int j = 0; j < rNew; j++)
        {
            for (int i = 0; i < this->m; i++)
                this->leftVectors[i + j*this->ldLeft] *= sqrt(s[j]);
            for (int i = 0; i < rhs.n; i++)
                this->rightVectors[i + j*this->ldRight] *= sqrt(s[j]);
        }
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0, workopt, *work;
            int info;
            
            // matrix multiply
            int ldr = this->ldRight, ldl = rhs.ldLeft;
            dgemm(&transA, &transB, &rLeft, &rRight, dimRhs, &alpha, (double*)(this->rightVectors),
                  &ldr, (double*)(rhs.leftVectors), &ldl, &beta, (double*)innerMat, &rLeft);
            
            // compute SVD
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)innerMat, &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)(innerMat), &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   work, &lwork, &info);
            mxFree(work);
            
            // populate vectors of new instance
            transA = 'N';
            transB = 'N';
            ldl = this->ldLeft;
            ldr = this->m;
            T* newLeft = (T*)mxMalloc(sizeof(T) * this->m * rNew);
            dgemm(&transA, &transB, &ldr, &rNew, &rLeft, &alpha,
                  (double*)(this->leftVectors), &ldl, (double*)innerMat, &rLeft,
                  &beta, (double*)newLeft, &ldr);
            // swap in new array
            mxFree(this->leftVectors);
            this->leftVectors = newLeft;
            
            transB = 'C';
            ldl = rhs.ldRight;
            ldr = rhs.n;
            T* newRight = (T*)mxMalloc(sizeof(T) * rhs.n * rNew);
            dgemm(&transA, &transB, &ldr, &rNew, &rRight, &alpha,
                  (double*)(rhs.rightVectors), &ldl, (double*)(innerMat + rLeft*rRight), &rLeft,
                  &beta, (double*)(newRight), &ldr);
            // swap in new array
            mxFree(this->rightVectors);
            this->rightVectors = newRight;
            
            // update storage info
            this->n = rhs.n;
            this->ldLeft = this->m;
            this->ldRight = this->n;
            this->rank = rNew;
            
            // modify new vectors
            for (int j = 0; j < rNew; j++)
            {
                for (int i = 0; i < this->m; i++)
                    this->leftVectors[i + j*this->ldLeft] *= sqrt(s[j]);
                for (int i = 0; i < rhs.n; i++)
                    this->rightVectors[i + j*this->ldRight] *= sqrt(s[j]);
            }
        }
        else
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:TimesEqual:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
    }
    
    mxFree(s);
    mxFree(innerMat);

    return (*this);
}

// multiply two low-rank matrices to form a new instance
template <class T> LowRankMat<T> LowRankMat<T>::operator * (const LowRankMat<T> rhs) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if (this->n != dimRhs[0])
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Times:dimMismatch",
                          "Inner matrix dimensions are not equal.");
        
    // perform matrix multiplication and svd
    int rLeft = this->rank, rRight = rhs.rank;
    int rNew = imin(rLeft, rRight);
    T* innerMat = (T*)mxMalloc(sizeof(T) * rLeft * rRight * 2);
    char transA = 'C', transB = 'N';
    char jobu = 'O', jobvt = 'S';
    double *s = (double*)mxMalloc(sizeof(double) * rNew);
    int lwork = -1;
    
    // create new instance
    LowRankMat<T> lhs(this->m, this->m, rhs.n, rhs.n, rNew);
    
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0, workopt, *work;
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * rNew);
        int info;
        
        // matrix multiply
        int ldr = this->ldRight, ldl = rhs.ldLeft;
        zgemm(&transA, &transB, &rLeft, &rRight, dimRhs, CCAST(&alpha), CCAST(this->rightVectors),
              &ldr, CCAST(rhs.leftVectors), &ldl, CCAST(&beta),
              CCAST(innerMat), &rLeft);
        
        // compute SVD
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // populate vectors of new instance
        transA = 'N';
        transB = 'N';
        ldl = this->ldLeft;
        ldr = this->m;
        zgemm(&transA, &transB, &ldr, &rNew, &rLeft, CCAST(&alpha),
              CCAST(this->leftVectors), &ldl, CCAST(innerMat), &rLeft,
              CCAST(&beta), CCAST(lhs.leftVectors), &ldr);
        transB = 'C';
        ldl = rhs.ldRight;
        ldr = rhs.n;
        zgemm(&transA, &transB, &ldr, &rNew, &rRight, CCAST(&alpha),
              CCAST(rhs.rightVectors), &ldl, CCAST(innerMat + rLeft*rRight), &rLeft,
              CCAST(&beta), CCAST(lhs.rightVectors), &ldr);
        
        // modify new vectors
        for (int j = 0; j < rNew; j++)
        {
            for (int i = 0; i < this->m; i++)
                lhs.leftVectors[i + j*this->m] *= sqrt(s[j]);
            for (int i = 0; i < rhs.n; i++)
                lhs.rightVectors[i + j*rhs.n] *= sqrt(s[j]);
        }
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0, workopt, *work;
            int info;
            
            // matrix multiply
            int ldr = this->ldRight, ldl = rhs.ldLeft;
            dgemm(&transA, &transB, &rLeft, &rRight, dimRhs, &alpha, (double*)(this->rightVectors),
                  &ldr, (double*)(rhs.leftVectors), &ldl, &beta, (double*)innerMat, &rLeft);
            
            // compute SVD
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)innerMat, &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)(innerMat), &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   work, &lwork, &info);
            mxFree(work);
            
            // populate vectors of new instance
            transA = 'N';
            transB = 'N';
            ldl = this->ldLeft;
            ldr = this->m;
            dgemm(&transA, &transB, &ldr, &rNew, &rLeft, &alpha,
                  (double*)(this->leftVectors), &ldl, (double*)innerMat, &rLeft,
                  &beta, (double*)(lhs.leftVectors), &ldr);
            transB = 'C';
            ldl = rhs.ldRight;
            ldr = rhs.n;
            dgemm(&transA, &transB, &ldr, &rNew, &rRight, &alpha,
                  (double*)(rhs.rightVectors), &ldl, (double*)(innerMat + rLeft*rRight), &rLeft,
                  &beta, (double*)(lhs.rightVectors), &ldr);
            
            // modify new vectors
            for (int j = 0; j < rNew; j++)
            {
                for (int i = 0; i < this->m; i++)
                    lhs.leftVectors[i + j*this->m] *= sqrt(s[j]);
                for (int i = 0; i < rhs.n; i++)
                    lhs.rightVectors[i + j*rhs.n] *= sqrt(s[j]);
            }
        }
        else
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:Times:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
    }
    
    mxFree(s);
    mxFree(innerMat);
    
    return lhs;
}

/************* specialized addition *************/

// add and assign a submatrix of another low-rank matrix
template <class T> void LowRankMat<T>::AddAssignPart(LowRankMat<T> *rhs, const int rI, const int cI)
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if ((rI < 0) || ((dimRhs[0] - rI) < this->m))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:AddAssignPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if ((cI < 0) || ((dimRhs[1] - cI) < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:AddAssignPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
        
    // get augmented rank
    int ro = rhs->rank;
    
    // reallocate to include space for new data
    int newNumel = this->ldLeft * (this->rank + ro);
    this->leftVectors = (T*)mxRealloc(this->leftVectors, sizeof(T) * newNumel);
    newNumel = this->ldRight * (this->rank + ro);
    this->rightVectors = (T*)mxRealloc(this->rightVectors, sizeof(T) * newNumel);
    
    // add in new data to each side
    rhs->CopyData(this->leftVectors + this->rank * this->ldLeft, this->ldLeft,
                  this->rightVectors + this->rank * this->ldRight, this->ldRight,
                  rI, cI, this->m, this->n);
    this->rank += ro;
    
    // compress
    this->CompressVectors();
}

// subtract and assign a submatrix of another low-rank matrix
template <class T> void LowRankMat<T>::SubAssignPart(LowRankMat<T> *rhs, const int rI, const int cI)
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if ((rI < 0) || ((dimRhs[0] - rI) < this->m))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SubAssignPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if ((cI < 0) || ((dimRhs[1] - cI) < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SubAssignPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
        
    // get augmented rank
    int r1 = rhs->Rank();
    
    // reallocate to include space for new data
    int newNumel = this->ldLeft * (this->rank + r1);
    this->leftVectors = (T*)mxRealloc(this->leftVectors, sizeof(T) * newNumel);
    newNumel = this->ldRight * (this->rank + r1);
    this->rightVectors = (T*)mxRealloc(this->rightVectors, sizeof(T) * newNumel);
    
    // add in new data to each side
    rhs->CopyData(this->leftVectors + this->rank * this->ldLeft, this->ldLeft,
                 this->rightVectors + this->rank * this->ldRight, this->ldRight,
                 rI, cI, this->m, this->n);
    int oRank = this->rank;
    this->rank += r1;
    this->Negate(oRank, r1);
    
    // compress
    this->CompressVectors();
}

// add a low-rank matrix with a submatrix of another low-rank matrix
template <class T> LowRankMat<T> LowRankMat<T>::AddPart(LowRankMat<T> *rhs, const int rI, const int cI) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if ((rI < 0) || ((dimRhs[0] - rI) < this->m))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:AddPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if ((cI < 0) || ((dimRhs[1] - cI) < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:AddPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
        
    // get new rank
    int rnew = rhs->rank + this->rank;
    
    // create new instance
    LowRankMat<T> lhs(this->m, this->m, this->n, this->n, rnew);
    
    // copy in vectors
    lhs.SetVectors(this->leftVectors, this->ldLeft, this->rightVectors, this->ldRight, 0, this->rank);
    lhs.SetVectors(rhs->leftVectors + rI, rhs->ldLeft, rhs->rightVectors + cI, rhs->ldRight,
                   this->rank, rhs->rank);
    
    // compress representation
    lhs.CompressVectors();
    
    return lhs;
}

// subtract a low-rank matrix with a submatrix of another low-rank matrix
template <class T> LowRankMat<T> LowRankMat<T>::SubPart(LowRankMat<T> *rhs, const int rI, const int cI) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if ((rI < 0) || ((dimRhs[0] - rI) < this->m))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SubPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if ((cI < 0) || ((dimRhs[1] - cI) < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:SubPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
    
    // get new rank
    int rnew = rhs->Rank() + this->rank;
    
    // create new instance
    LowRankMat<T> lhs(this->m, this->m, this->n, this->n, rnew);
    
    // copy in vectors
    lhs.SetVectors(this->leftVectors, this->ldLeft, this->rightVectors, this->ldRight, 0, this->rank);
    lhs.SetVectors(rhs->leftVectors + rI, rhs->ldLeft, rhs->rightVectors + cI, rhs->ldRight,
                   this->rank, rhs->rank);
    lhs.Negate(this->rank, rhs->rank);
    
    // compress representation
    lhs.CompressVectors();
    
    return lhs;
}

/************* specialized multiplication *************/

// right-multiply and assign by a submatrix of another low-rank matrix
template <class T> void LowRankMat<T>::RMultAssignPart(LowRankMat<T> *rhs, const int rI, const int cI, const int nCols)
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if ((rI < 0) || ((dimRhs[0] - rI) < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultAssignPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultAssignPart:dimMismatch",
                          "Specified number of columns to use is invalid.");
    }  
    if ((cI < 0) || ((dimRhs[1] - cI) < nCols))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultAssignPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
    
    // perform matrix multiplication and svd
    int rLeft = this->rank, rRight = rhs->rank;
    int rNew = imin(rLeft, rRight);
    T* innerMat = (T*)mxMalloc(sizeof(T) * rLeft * rRight * 2);
    char transA = 'C', transB = 'N';
    char jobu = 'O', jobvt = 'S';
    double *s = (double*)mxMalloc(sizeof(double) * rNew);
    int lwork = -1;
    int newDim = nCols;
    int innerDim = this->n;
    
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0, workopt, *work;
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * rNew);
        int info;
        
        // matrix multiply
        int ldr = this->ldRight, ldl = rhs->ldLeft;
        
        zgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
              CCAST(&alpha), CCAST(this->rightVectors), &ldr,
              CCAST(rhs->leftVectors + rI), &ldl, CCAST(&beta),
              CCAST(innerMat), &rLeft);
        
        // compute SVD
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // populate vectors of new instance
        transA = 'N';
        transB = 'N';
        ldl = this->ldLeft;
        ldr = this->m;
        T* newLeft = (T*)mxMalloc(sizeof(T) * this->m * rNew);
        zgemm(&transA, &transB, &ldr, &rNew, &rLeft, CCAST(&alpha),
              CCAST(this->leftVectors), &ldl, CCAST(innerMat), &rLeft,
              CCAST(&beta), CCAST(newLeft), &ldr);
        // swap in new array
        mxFree(this->leftVectors);
        this->leftVectors = newLeft;
        
        transB = 'C';
        ldl = rhs->ldRight;
        ldr = newDim;
        T* newRight = (T*)mxMalloc(sizeof(T) * ldr * rNew);
        zgemm(&transA, &transB, &ldr, &rNew, &rRight, CCAST(&alpha),
              CCAST(rhs->rightVectors + cI), &ldl, CCAST(innerMat + rLeft*rRight), &rLeft,
              CCAST(&beta), CCAST(newRight), &ldr);
        // swap in new array
        mxFree(this->rightVectors);
        this->rightVectors = newRight;
        
        // update storage info
        this->n = ldr;
        this->ldLeft = this->m;
        this->ldRight = ldr;
        this->rank = rNew;
        
        // modify new vectors
        for (int j = 0; j < rNew; j++)
        {
            for (int i = 0; i < this->m; i++)
                this->leftVectors[i + j*this->ldLeft] *= sqrt(s[j]);
            for (int i = 0; i < newDim; i++)
                this->rightVectors[i + j*this->ldRight] *= sqrt(s[j]);
        }
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0, workopt, *work;
            int info;
            
            // matrix multiply
            int ldr = this->ldRight, ldl = rhs->ldLeft;
            dgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
                  &alpha, (double*)(this->rightVectors), &ldr,
                  (double*)(rhs->leftVectors + rI), &ldl, &beta, (double*)innerMat, &rLeft);
            
            // compute SVD
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)innerMat, &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)(innerMat), &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   work, &lwork, &info);
            mxFree(work);
            
            // populate vectors of new instance
            transA = 'N';
            transB = 'N';
            ldl = this->ldLeft;
            ldr = this->m;
            T* newLeft = (T*)mxMalloc(sizeof(T) * this->m * rNew);
            dgemm(&transA, &transB, &ldr, &rNew, &rLeft, &alpha,
                  (double*)(this->leftVectors), &ldl, (double*)innerMat, &rLeft,
                  &beta, (double*)newLeft, &ldr);
            // swap in new array
            mxFree(this->leftVectors);
            this->leftVectors = newLeft;
            
            transB = 'C';
            ldl = rhs->ldRight;
            ldr = newDim;
            T* newRight = (T*)mxMalloc(sizeof(T) * ldr * rNew);
            dgemm(&transA, &transB, &ldr, &rNew, &rRight, &alpha,
                  (double*)(rhs->rightVectors + cI), &ldl, (double*)(innerMat + rLeft*rRight), &rLeft,
                  &beta, (double*)(newRight), &ldr);
            // swap in new array
            mxFree(this->rightVectors);
            this->rightVectors = newRight;
            
            // update storage info
            this->n = ldr;
            this->ldLeft = this->m;
            this->ldRight = ldr;
            this->rank = rNew;
            
            // modify new vectors
            for (int j = 0; j < rNew; j++)
            {
                for (int i = 0; i < this->m; i++)
                    this->leftVectors[i + j*this->ldLeft] *= sqrt(s[j]);
                for (int i = 0; i < newDim; i++)
                    this->rightVectors[i + j*this->ldRight] *= sqrt(s[j]);
            }
        }
        else
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultAssignPart:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
    }
    mxFree(s);
    mxFree(innerMat);
}

// right-multiply and assign by a dense matrix
template <class T> void LowRankMat<T>::RMultAssignPart(T *rhs, const int nCols, const int ldim)
{
    // check input column
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultAssignPart:badNumCols",
                          "Specified number of columns is invalid.");
    }  
    if (ldim < this->n)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultAssignPart:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    char transA = 'C', transB = 'N';
    // perform matrix multiplication
    int numCols = nCols;
    int ldl = ldim;
    int ldr = this->ldRight;
    T* newRight = (T*)mxMalloc(sizeof(T) * this->rank * nCols);
    if (sizeof(T) == sizeof(cmpx))
    {
        // matrix multiply
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &numCols, &(this->rank), &(this->n),
              CCAST(&alpha), CCAST(rhs), &ldl, CCAST(this->rightVectors), &ldr,
              CCAST(&beta), CCAST(newRight), &numCols);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            // matrix multiply
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &numCols, &(this->rank), &(this->n),
                  &alpha, (double*)(rhs), &ldl, (double*)(this->rightVectors), &ldr,
                  &beta, (double*)newRight, &numCols);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultAssignPart:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
            return;
        }
    }
    mxFree(this->rightVectors);
    this->rightVectors = newRight;
    this->ldRight = numCols;
    this->n = numCols;
}

// left-multiply and assign by a submatrix of another low-rank matrix
template <class T> void LowRankMat<T>::LMultAssignPart(LowRankMat<T> *rhs, const int rI, const int cI, const int nRows)
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if (nRows < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultAssignPart:dimMismatch",
                          "Specified number of rows to use is invalid.");
    }
    if ((rI < 0) || ((dimRhs[0] - rI) < nRows))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultAssignPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if ((cI < 0) || ((dimRhs[1] - cI) < this->m))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultAssignPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
    
    // perform matrix multiplication and svd
    int rLeft = rhs->rank, rRight = this->rank;
    int rNew = imin(rLeft, rRight);
    T* innerMat = (T*)mxMalloc(sizeof(T) * rLeft * rRight * 2);
    char transA = 'C', transB = 'N';
    char jobu = 'O', jobvt = 'S';
    double *s = (double*)mxMalloc(sizeof(double) * rNew);
    int lwork = -1;
    int newDim = nRows;
    int innerDim = this->m;
    
    // complex data
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0, workopt, *work;
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * rNew);
        int info;
        
        // matrix multiply
        int ldr = rhs->ldRight, ldl = this->ldLeft;
        zgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
              CCAST(&alpha), CCAST(rhs->rightVectors + cI), &ldr,
              CCAST(this->leftVectors), &ldl, CCAST(&beta),
              CCAST(innerMat), &rLeft);
        
        // compute SVD
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // populate vectors of new instance
        transA = 'N';
        transB = 'N';
        ldl = rhs->ldLeft;
        ldr = newDim;
        T* newLeft = (T*)mxMalloc(sizeof(T) * ldr * rNew);
        zgemm(&transA, &transB, &ldr, &rNew, &rLeft, CCAST(&alpha),
              CCAST(rhs->leftVectors + rI), &ldl, CCAST(innerMat), &rLeft,
              CCAST(&beta), CCAST(newLeft), &ldr);
        // swap in new array
        mxFree(this->leftVectors);
        this->leftVectors = newLeft;
        
        transB = 'C';
        ldl = this->ldRight;
        ldr = this->n;
        T* newRight = (T*)mxMalloc(sizeof(T) * ldr * rNew);
        zgemm(&transA, &transB, &ldr, &rNew, &rRight, CCAST(&alpha),
              CCAST(this->rightVectors), &ldl, CCAST(innerMat + rLeft*rRight), &rLeft,
              CCAST(&beta), CCAST(newRight), &ldr);
        // swap in new array
        mxFree(this->rightVectors);
        this->rightVectors = newRight;
        
        // update storage info
        this->m = newDim;
        this->ldLeft = newDim;
        this->ldRight = this->n;
        this->rank = rNew;
        
        // modify new vectors
        for (int j = 0; j < rNew; j++)
        {
            for (int i = 0; i < this->m; i++)
                this->leftVectors[i + j*this->ldLeft] *= sqrt(s[j]);
            for (int i = 0; i < this->n; i++)
                this->rightVectors[i + j*this->ldRight] *= sqrt(s[j]);
        }
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0, workopt, *work;
            int info;
            
            // matrix multiply
            int ldr = rhs->ldRight, ldl = this->ldLeft;
            dgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
                  &alpha, (double*)(rhs->rightVectors + cI), &ldr,
                  (double*)(this->leftVectors), &ldl, &beta, (double*)innerMat, &rLeft);
            
            // compute SVD
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)innerMat, &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)(innerMat), &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   work, &lwork, &info);
            mxFree(work);
            
            // populate vectors of new instance
            transA = 'N';
            transB = 'N';
            ldl = rhs->ldLeft;
            ldr = newDim;
            T* newLeft = (T*)mxMalloc(sizeof(T) * ldr * rNew);
            dgemm(&transA, &transB, &ldr, &rNew, &rLeft, &alpha,
                  (double*)(rhs->leftVectors + rI), &ldl, (double*)innerMat, &rLeft,
                  &beta, (double*)newLeft, &ldr);
            // swap in new array
            mxFree(this->leftVectors);
            this->leftVectors = newLeft;
            
            transB = 'C';
            ldl = this->ldRight;
            ldr = this->n;
            T* newRight = (T*)mxMalloc(sizeof(T) * ldr * rNew);
            dgemm(&transA, &transB, &ldr, &rNew, &rRight,
                  &alpha, (double*)(this->rightVectors), &ldl,
                  (double*)(innerMat + rLeft*rRight), &rLeft, &beta,
                  (double*)(newRight), &ldr);
            // swap in new array
            mxFree(this->rightVectors);
            this->rightVectors = newRight;
            
            // update storage info
            this->m = newDim;
            this->ldLeft = newDim;
            this->ldRight = this->n;
            this->rank = rNew;
            
            // modify new vectors
            for (int j = 0; j < rNew; j++)
            {
                for (int i = 0; i < this->m; i++)
                    this->leftVectors[i + j*this->ldLeft] *= sqrt(s[j]);
                for (int i = 0; i < this->n; i++)
                    this->rightVectors[i + j*this->ldRight] *= sqrt(s[j]);
            }
        }
        else
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultAssignPart:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
    }
    mxFree(s);
    mxFree(innerMat);
}

// left-multiply and assign by a dense matrix
template <class T> void LowRankMat<T>::LMultAssignPart(T *rhs, const int nRows, const int ldim)
{
    // check input column
    if (nRows < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultAssignPart:badNumCols",
                          "Specified number of rows is invalid.");
    }  
    if (ldim < nRows)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultAssignPart:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    char transA = 'N', transB = 'N';
    // perform matrix multiplication
    int numRows = nRows;
    int ldl = ldim;
    int ldr = this->ldLeft;
    T* newLeft = (T*)mxMalloc(sizeof(T) * this->rank * nRows);
    if (sizeof(T) == sizeof(cmpx))
    {
        // matrix multiply
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &numRows, &(this->rank), &(this->m),
              CCAST(&alpha), CCAST(rhs), &ldl, CCAST(this->leftVectors), &ldr,
              CCAST(&beta), CCAST(newLeft), &numRows);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            // matrix multiply
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &numRows, &(this->rank), &(this->m),
                  &alpha, (double*)(rhs), &ldl, (double*)(this->leftVectors), &ldr,
                  &beta, (double*)newLeft, &numRows);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultAssignPart:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
            return;
        }
    }
    mxFree(this->leftVectors);
    this->leftVectors = newLeft;
    this->ldLeft = numRows;
    this->m = numRows;
}

// right-multiply by a submatrix of another low-rank matrix
template <class T> LowRankMat<T>* LowRankMat<T>::RMultPart(LowRankMat<T> *rhs, const int rI, const int cI, const int nCols) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if ((rI < 0) || ((dimRhs[0] - rI) < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultPart:dimMismatch",
                          "Specified number of columns to use is invalid.");
    }  
    if ((cI < 0) || ((dimRhs[1] - cI) < nCols))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
    
    // perform matrix multiplication and svd
    int rLeft = this->rank, rRight = rhs->rank;
    int rNew = imin(rLeft, rRight);
    T* innerMat = (T*)mxMalloc(sizeof(T) * rLeft * rRight * 2);
    char transA = 'C', transB = 'N';
    char jobu = 'O', jobvt = 'S';
    double *s = (double*)mxMalloc(sizeof(double) * rNew);
    int lwork = -1;
    int newDim = nCols;
    int innerDim = this->n;
    
    // create new instance
    LowRankMat<T>* lhs = new LowRankMat<T>(this->m, this->m, newDim, newDim, rNew);
    
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0, workopt, *work;
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * rNew);
        int info;
        
        // matrix multiply
        int ldr = this->ldRight, ldl = rhs->ldLeft;
        
        zgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
              CCAST(&alpha), CCAST(this->rightVectors), &ldr,
              CCAST(rhs->leftVectors + rI), &ldl, CCAST(&beta),
              CCAST(innerMat), &rLeft);
        
        // compute SVD
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // populate vectors of new instance
        transA = 'N';
        transB = 'N';
        ldl = this->ldLeft;
        ldr = this->m;
        zgemm(&transA, &transB, &ldr, &rNew, &rLeft, CCAST(&alpha),
              CCAST(this->leftVectors), &ldl, CCAST(innerMat), &rLeft,
              CCAST(&beta), CCAST(lhs->leftVectors), &ldr);
        transB = 'C';
        ldl = rhs->ldRight;
        ldr = newDim;
        zgemm(&transA, &transB, &ldr, &rNew, &rRight, CCAST(&alpha),
              CCAST(rhs->rightVectors + cI), &ldl, CCAST(innerMat + rLeft*rRight), &rLeft,
              CCAST(&beta), CCAST(lhs->rightVectors), &ldr);
        
        // modify new vectors
        for (int j = 0; j < rNew; j++)
        {
            for (int i = 0; i < this->m; i++)
                lhs->leftVectors[i + j*this->m] *= sqrt(s[j]);
            for (int i = 0; i < newDim; i++)
                lhs->rightVectors[i + j*newDim] *= sqrt(s[j]);
        }
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0, workopt, *work;
            int info;
            
            // matrix multiply
            int ldr = this->ldRight, ldl = rhs->ldLeft;
            dgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
                  &alpha, (double*)(this->rightVectors), &ldr,
                  (double*)(rhs->leftVectors + rI), &ldl, &beta, (double*)innerMat, &rLeft);
            
            // compute SVD
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)innerMat, &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)(innerMat), &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   work, &lwork, &info);
            mxFree(work);
            
            // populate vectors of new instance
            transA = 'N';
            transB = 'N';
            ldl = this->ldLeft;
            ldr = this->m;
            dgemm(&transA, &transB, &ldr, &rNew, &rLeft, &alpha,
                  (double*)(this->leftVectors), &ldl, (double*)innerMat, &rLeft,
                  &beta, (double*)(lhs->leftVectors), &ldr);
            transB = 'C';
            ldl = rhs->ldRight;
            ldr = newDim;
            dgemm(&transA, &transB, &ldr, &rNew, &rRight, &alpha,
                  (double*)(rhs->rightVectors + cI), &ldl, (double*)(innerMat + rLeft*rRight), &rLeft,
                  &beta, (double*)(lhs->rightVectors), &ldr);
            
            // modify new vectors
            for (int j = 0; j < rNew; j++)
            {
                for (int i = 0; i < this->m; i++)
                    lhs->leftVectors[i + j*this->m] *= sqrt(s[j]);
                for (int i = 0; i < newDim; i++)
                    lhs->rightVectors[i + j*newDim] *= sqrt(s[j]);
            }
        }
        else
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultPart:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
    }
    
    mxFree(s);
    mxFree(innerMat);
    
    return lhs;
}

// right-multiply by a dense matrix
template <class T> LowRankMat<T>* LowRankMat<T>::RMultPart(T* rhs, const int nCols, const int ldim) const
{
    LowRankMat<T> *lhs = new LowRankMat<T>(*this);
    lhs->RMultAssignPart(rhs, nCols, ldim);
    return lhs;
}

// left-multiply by a dense matrix
template <class T> LowRankMat<T>* LowRankMat<T>::LMultPart(T* rhs, const int nRows, const int ldim) const
{
    LowRankMat<T> *lhs = new LowRankMat<T>(*this);
    lhs->LMultAssignPart(rhs, nRows, ldim);
    return lhs;
}

// left-multiply by a submatrix of another low-rank matrix
template <class T> LowRankMat<T>* LowRankMat<T>::LMultPart(LowRankMat<T> *rhs, const int rI, const int cI, const int nRows) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check input column and row indices
    if (nRows < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultPart:dimMismatch",
                          "Specified number of rows to use is invalid.");
    }
    if ((rI < 0) || ((dimRhs[0] - rI) < nRows))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultPart:dimMismatch",
                          "Supplied row index is out of range.");
    }
    if ((cI < 0) || ((dimRhs[1] - cI) < this->m))
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultPart:dimMismatch",
                          "Supplied column index is out of range.");
    }
    
    // perform matrix multiplication and svd
    int rLeft = rhs->rank, rRight = this->rank;
    int rNew = imin(rLeft, rRight);
    T* innerMat = (T*)mxMalloc(sizeof(T) * rLeft * rRight * 2);
    char transA = 'C', transB = 'N';
    char jobu = 'O', jobvt = 'S';
    double *s = (double*)mxMalloc(sizeof(double) * rNew);
    int lwork = -1;
    int newDim = nRows;
    int innerDim = this->m;
    
    // create new instance
    LowRankMat<T>* lhs = new LowRankMat<T>(newDim, newDim, this->n, this->n, rNew);
    
    // complex data
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0, workopt, *work;
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * rNew);
        int info;
        
        // matrix multiply
        int ldr = rhs->ldRight, ldl = this->ldLeft;
        zgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
              CCAST(&alpha), CCAST(rhs->rightVectors + cI), &ldr,
              CCAST(this->leftVectors), &ldl, CCAST(&beta),
              CCAST(innerMat), &rLeft);
        
        // compute SVD
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &rLeft, &rRight, CCAST(innerMat), &rLeft, s,
               CCAST(innerMat), &rLeft, CCAST(innerMat + rLeft * rRight), &rLeft,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // populate vectors of new instance
        transA = 'N';
        transB = 'N';
        ldl = rhs->ldLeft;
        ldr = newDim;
        zgemm(&transA, &transB, &ldr, &rNew, &rLeft, CCAST(&alpha),
              CCAST(rhs->leftVectors + rI), &ldl, CCAST(innerMat), &rLeft,
              CCAST(&beta), CCAST(lhs->leftVectors), &newDim);
        transB = 'C';
        ldl = this->ldRight;
        ldr = this->n;
        zgemm(&transA, &transB, &ldr, &rNew, &rRight, CCAST(&alpha),
              CCAST(this->rightVectors), &ldl, CCAST(innerMat + rLeft*rRight), &rLeft,
              CCAST(&beta), CCAST(lhs->rightVectors), &ldr);
        
        // modify new vectors
        for (int j = 0; j < rNew; j++)
        {
            for (int i = 0; i < newDim; i++)
                lhs->leftVectors[i + j*newDim] *= sqrt(s[j]);
            for (int i = 0; i < this->n; i++)
                lhs->rightVectors[i + j*this->n] *= sqrt(s[j]);
        }
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0, workopt, *work;
            int info;
            
            // matrix multiply
            int ldr = rhs->ldRight, ldl = this->ldLeft;
            dgemm(&transA, &transB, &rLeft, &rRight, &innerDim,
                  &alpha, (double*)(rhs->rightVectors + cI), &ldr,
                  (double*)(this->leftVectors), &ldl, &beta, (double*)innerMat, &rLeft);
            
            // compute SVD
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)innerMat, &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgesvd(&jobu, &jobvt, &rLeft, &rRight, (double*)(innerMat), &rLeft, s,
                   (double*)innerMat, &rLeft, (double*)(innerMat + rLeft * rRight), &rLeft,
                   work, &lwork, &info);
            mxFree(work);
            
            // populate vectors of new instance
            transA = 'N';
            transB = 'N';
            ldl = rhs->ldLeft;
            ldr = newDim;
            dgemm(&transA, &transB, &ldr, &rNew, &rLeft, &alpha,
                  (double*)(rhs->leftVectors + rI), &ldl, (double*)innerMat, &rLeft,
                  &beta, (double*)(lhs->leftVectors), &newDim);
            transB = 'C';
            ldl = this->ldRight;
            ldr = this->n;
            dgemm(&transA, &transB, &ldr, &rNew, &rRight, &alpha,
                  (double*)(this->rightVectors), &ldl, (double*)(innerMat + rLeft*rRight), &rLeft,
                  &beta, (double*)(lhs->rightVectors), &ldr);
            
            // modify new vectors
            for (int j = 0; j < rNew; j++)
            {
                for (int i = 0; i < newDim; i++)
                    lhs->leftVectors[i + j*newDim] *= sqrt(s[j]);
                for (int i = 0; i < this->n; i++)
                    lhs->rightVectors[i + j*this->n] *= sqrt(s[j]);
            }
        }
        else
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultPart:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
    }
    
    mxFree(s);
    mxFree(innerMat);

    return lhs;
}

// right-multiply by a dense matrix, returning a dense matrix
template <class T> T* LowRankMat<T>::RMultDense(T* rhs, const int nCols, const int ldim, int& ldout) const
{
    // check input column
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultDense:badNumCols",
                          "Specified number of columns is invalid.");
    }  
    if (ldim < this->n)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultDense:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    char transA = 'C', transB = 'N';
    // perform matrix multiplication
    int numCols = nCols;
    int ldl = ldim;
    int ldr = this->ldRight;
    T* temp = (T*)mxMalloc(sizeof(T) * this->rank * nCols);
    int rk = this->rank, innerDim = this->n;
    if (sizeof(T) == sizeof(cmpx))
    {
        // matrix multiply
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &numCols, &rk, &innerDim,
              CCAST(&alpha), CCAST(rhs), &ldl, CCAST(this->rightVectors), &ldr,
              CCAST(&beta), CCAST(temp), &numCols);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            // matrix multiply
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &numCols, &rk, &innerDim,
                  &alpha, (double*)(rhs), &ldl, (double*)(this->rightVectors), &ldr,
                  &beta, (double*)temp, &numCols);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultDense:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
            mxFree(temp);
            return NULL;
        }
    }
    
    T* lhs = (T*)mxMalloc(sizeof(T) * this->m * numCols);
    transA = 'N';
    transB = 'C';
    ldl = this->ldLeft;
    ldr = this->m;
    if (sizeof(T) == sizeof(cmpx))
    {
        // matrix multiply
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &ldr, &numCols, &rk,
              CCAST(&alpha), CCAST(this->leftVectors), &ldl, CCAST(temp), &numCols,
              CCAST(&beta), CCAST(lhs), &ldr);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            // matrix multiply
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &ldr, &numCols, &rk,
                  &alpha, (double*)(this->leftVectors), &ldl, (double*)(temp), &numCols,
                  &beta, (double*)lhs, &ldr);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:RMultDense:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
            mxFree(temp);
            mxFree(lhs);
            return NULL;
        }
    }
    ldout = this->m;
    mxFree(temp);
    return lhs;
}

// left-multiply by a dense matrix, returning a dense matrix
template <class T> T* LowRankMat<T>::LMultDense(T* rhs, const int nRows, const int ldim, int& ldout) const
{
    // check input column
    if (nRows < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultDense:badNumCols",
                          "Specified number of rows is invalid.");
    }  
    if (ldim < nRows)
    {
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultDense:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    char transA = 'N', transB = 'N';
    // perform matrix multiplication
    int numRows = nRows;
    int ldl = ldim;
    int ldr = this->ldLeft;
    T* temp = (T*)mxMalloc(sizeof(T) * nRows * this->rank);
    int rk = this->rank, innerDim = this->m;
    if (sizeof(T) == sizeof(cmpx))
    {
        // matrix multiply
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &numRows, &rk, &innerDim,
              CCAST(&alpha), CCAST(rhs), &ldl, CCAST(this->leftVectors), &ldr,
              CCAST(&beta), CCAST(temp), &numRows);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            // matrix multiply
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &numRows, &rk, &innerDim,
                  &alpha, (double*)(rhs), &ldl, (double*)(this->leftVectors), &ldr,
                  &beta, (double*)temp, &numRows);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultDense:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
            mxFree(temp);
            return NULL;
        }
    }
    
    T* lhs = (T*)mxMalloc(sizeof(T) * this->n * numRows);
    transA = 'N';
    transB = 'C';
    ldl = numRows;
    ldr = this->ldRight;
    int outerDim = this->n;
    if (sizeof(T) == sizeof(cmpx))
    {
        // matrix multiply
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &numRows, &outerDim, &rk,
              CCAST(&alpha), CCAST(temp), &ldl, CCAST(this->rightVectors), &ldr,
              CCAST(&beta), CCAST(lhs), &numRows);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            // matrix multiply
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &numRows, &outerDim, &rk,
                  &alpha, (double*)temp, &ldl, (double*)(this->rightVectors), &ldr,
                  &beta, (double*)lhs, &numRows);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:LMultDense:invalidDataType",
                               "Cannot multiply because data type is not double or cmpx.");
            mxFree(temp);
            mxFree(lhs);
            return NULL;
        }
    }
    mxFree(temp);
    ldout = numRows;
    return lhs;
}

/************* other *************/

// make a dense version
template <class T> void LowRankMat<T>::MakeDense(T** src, int &ldim, const int rI=0, const int cI=0)
{
    // check to make sure input array is OK
    if (src == NULL)
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:nullData",
                          "Input data location appears to be NULL.");
    
    // check to make sure data is non-null
    if ((this->leftVectors == NULL) || (this->rightVectors == NULL))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:nullData",
                          "Internal data appears to be NULL.");
        
    // check index ranges
    if ((rI < 0) || (rI >= this->m) || (cI < 0) || (cI >= this->n))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:badIndices",
                          "Supplied submatrix indices are invalid.");
    
    // allocate dense array for data
    int mNew = this->m - rI, nNew = this->n - cI;
    ldim = mNew;
    T* dataCopy = (T*)mxMalloc(sizeof(T) * mNew * nNew);
    
    // create product
    char transA = 'N', transB = 'C';
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &mNew, &nNew, &(this->rank),
              CCAST(&alpha), CCAST(this->leftVectors + rI), &(this->ldLeft),
              CCAST(this->rightVectors + cI), &(this->ldRight), CCAST(&beta),
              CCAST(dataCopy), &mNew);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &mNew, &nNew, &(this->rank),
                  &alpha, (double*)(this->leftVectors + rI), &(this->ldLeft),
                  (double*)(this->rightVectors + cI), &(this->ldRight), &beta,
                  (double*)(dataCopy), &mNew);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:badDataType",
                          "Cannot make dense because data type is not double or cmpx.");
        }
    }
    (*src) = dataCopy;
}

// make a dense version
template <class T> void LowRankMat<T>::MakeDense(T** src, int &ldim, const int rI, const int cI,
                                                 const int nRows, const int nCols)
{
    // check to make sure input array is OK
    if (src == NULL)
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:nullData",
                          "Input data location appears to be NULL.");
        
    // check to make sure data is non-null
    if ((this->leftVectors == NULL) || (this->rightVectors == NULL))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:nullData",
                          "Internal data appears to be NULL.");
        
    // check index ranges
    if ((rI < 0) || ((this->m - rI) < nRows) ||
        (cI < 0) || ((this->n - cI) < nCols))
        mexErrMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:badIndices",
                          "Supplied submatrix indices are invalid.");
    
    // allocate dense array for data
    int mNew = nRows, nNew = nCols;
    ldim = mNew;
    T* dataCopy = (T*)mxMalloc(sizeof(T) * mNew * nNew);
    
    // create product
    char transA = 'N', transB = 'C';
    if (sizeof(T) == sizeof(cmpx))
    {
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &mNew, &nNew, &(this->rank),
              CCAST(&alpha), CCAST(this->leftVectors + rI), &(this->ldLeft),
              CCAST(this->rightVectors + cI), &(this->ldRight), CCAST(&beta),
              CCAST(dataCopy), &mNew);
    }
    else
    {
        if (sizeof(T) == sizeof(double))
        {
            double alpha = 1.0, beta = 0.0;
            dgemm(&transA, &transB, &mNew, &nNew, &(this->rank),
                  &alpha, (double*)(this->leftVectors + rI), &(this->ldLeft),
                  (double*)(this->rightVectors + cI), &(this->ldRight), &beta,
                  (double*)(dataCopy), &mNew);
        }
        else
        {
            mexWarnMsgIdAndTxt("MATLAB:lrmat:LowRankMat:MakeDense:badDataType",
                          "Cannot make dense because data type is not double or cmpx.");
        }
    }
    (*src) = dataCopy;
}