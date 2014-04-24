/*
    @(#)File:                /kernel/hmat.h
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        20 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Declares routines for hierarchical matrices
*/
#pragma once

#include <cmath>
#include <cstdio>
#include <complex>
#include <climits>
#include "mex.h"
#include "lrmat.h"
#include "misc.h"

#ifndef _RANKDEFS
#define _RANKDEFS
#define lowrank true
#define dense false
#endif
#ifndef _NMETA
#define _NMETA
#define NUMMETA 11
#endif

using namespace std;

/*  hierarchical matrix class */
template <class T> class HMat {

    public:
        
        // default initialization
        HMat()
        {
            this->northWest = NULL;
            this->northEast = NULL;
            this->southWest = NULL;
            this->southEast = NULL;
            this->data = NULL;
            this->ldim = 0;
            this->m = 0;
            this->n = 0;
            memset(this->dimSub, 0, sizeof(int) * 4);
        }
        
        // destructor
        ~HMat()
        {
            if (this->northWest != NULL)
            {
                delete this->northWest;
                this->northWest = NULL;
            }
            if (this->northEast != NULL)
            {
                delete this->northEast;
                this->northEast = NULL;
            }
            if (this->southWest != NULL)
            {
                delete this->southWest;
                this->southWest = NULL;
            }
            if (this->southEast != NULL)
            {
                delete this->southEast;
                this->southEast = NULL;
            }
            if (this->data != NULL)
            {
                mxFree(this->data);
                this->data = NULL;
            }
            this->ldim = 0;
            this->m = 0;
            this->n = 0;
            memset(this->dimSub, 0, sizeof(int) * 4);
        }
        
        // copy constructor
        HMat(HMat const& rhs)
        {
            this->northWest = NULL;
            this->southEast = NULL;
            this->southWest = NULL;
            this->northEast = NULL;
            this->data = NULL;
            this->ldim = 0;
            this->m = rhs.m;
            this->n = rhs.n;
            memset(this->dimSub, 0, sizeof(int) * 4);
                
            if (rhs.data != NULL)
            {
                if (rhs.ldim < rhs.m)
                {
                    mexErrMsgIdAndTxt("MATLAB:hmat:HMat:badInput",
                        "Stored leading dimension is less than the first matrix dimension.");
                }

                this->ldim = rhs.m;
                this->data = (T*)mxMalloc(sizeof(T) * rhs.m * rhs.n);
                for (int i = 0; i < rhs.n; i++)
                    memcpy(this->data + i*this->ldim, rhs.data + i*rhs.ldim, sizeof(T) * rhs.m);
                this->dimSub[0] = rhs.dimSub[0];
                this->dimSub[1] = rhs.dimSub[1];
            }
            else
            {
                if (rhs.northWest != NULL)
                    this->northWest = new HMat<T>(*(rhs.northWest));
                if (rhs.southEast != NULL)
                    this->southEast = new HMat<T>(*(rhs.southEast));
                if (rhs.northEast != NULL)
                    this->northEast = new LowRankMat<T>(*(rhs.northEast));
                if (rhs.southWest != NULL)
                    this->southWest = new LowRankMat<T>(*(rhs.southWest));
                memcpy(this->dimSub, rhs.dimSub, sizeof(int) * 4);
            }
        }
        
        // raw data constructor
        HMat(T* source, const int ldIn, const int mIn, const int nIn, const int lim)
        {
            this->northWest = NULL;
            this->northEast = NULL;
            this->southWest = NULL;
            this->southEast = NULL;
            this->data = NULL;
            this->ldim = 0;
            this->m = 0;
            this->n = 0;
            memset(this->dimSub, 0, sizeof(int) * 4);

            // input validation
            if (source == NULL)
                mexErrMsgIdAndTxt("MATLAB:hmat:HMat:badInput",
                    "Input data array is NULL.");
            if ((mIn < 1) || (nIn < 1))
                mexErrMsgIdAndTxt("MATLAB:hmat:HMat:badInput",
                    "Specified dimensions of input data array are invalid.");
            if (ldIn < mIn)
                mexErrMsgIdAndTxt("MATLAB:hmat:HMat:badInput",
                    "Specified leading dimension of data out of range.");
            if (lim < 1)
                mexErrMsgIdAndTxt("MATLAB:hmat:HMat:badInput",
                    "Bad limit specifier supplied.");
                
            this->m = mIn;
            this->n = nIn;
                
            // store matrix as dense
            if (imax(mIn, nIn) <= lim)
            {
                this->data = (T*)mxMalloc(sizeof(T) * mIn * nIn);
                this->ldim = mIn;
                for (int i = 0; i < nIn; i++)
                    memcpy(this->data + i*this->ldim, source + i*ldIn, sizeof(T) * mIn);
                dimSub[0] = this->m;
                dimSub[1] = this->n;
            }
            // store matrix as hierarchical
            else
            {
                int mh = (int)ceil(mIn / 2.0), nh = (int)ceil(nIn / 2.0);
                this->dimSub[0] = imin(mh, nh);
                this->dimSub[1] = this->dimSub[0];
                this->dimSub[2] = mIn - this->dimSub[0];
                this->dimSub[3] = nIn - this->dimSub[1];
                
                // create low-rank matrices
                this->northEast = new LowRankMat<T>(source + this->dimSub[1]*ldIn, ldIn,
                                                    this->dimSub[0], this->dimSub[3]);
                this->southWest = new LowRankMat<T>(source + this->dimSub[0], ldIn,
                                                    this->dimSub[2], this->dimSub[1]);
                // check ranks:
                int rankNE = (this->northEast)->Rank(), rankSW = (this->southWest)->Rank();
                // make dense if off-diagonal components aren't low rank
                if (((2*rankNE) > imin(this->dimSub[0], this->dimSub[3])) ||
                    ((2*rankSW) > imin(this->dimSub[1], this->dimSub[2])))
                {
                    delete (this->northEast);
                    this->northEast = NULL;
                    delete (this->southWest);
                    this->southWest = NULL;
                    memset(this->dimSub, 0, sizeof(int) * 4);
                    this->data = (T*)mxMalloc(sizeof(T) * mIn * nIn);
                    this->ldim = mIn;
                    for (int i = 0; i < nIn; i++)
                        memcpy(this->data + i*this->ldim, source + i*ldIn, sizeof(T) * mIn);
                    dimSub[0] = this->m;
                    dimSub[1] = this->n;
                }
                else
                {
                    // create hierarchical matrices
                    this->northWest = new HMat<T>(source, ldIn, this->dimSub[0], this->dimSub[1], lim);
                    this->southEast = new HMat<T>(source + this->dimSub[0] + this->dimSub[1]*ldIn, ldIn,
                                              this->dimSub[2], this->dimSub[3], lim);
                }
            }
        }
        
        // constructor from metadata
        HMat(T* dataIn, int* meta, const int dataLen, const int metaLen, int metaIdx=0, bool isValidated=false)
        {
            // initialize blank
            this->northWest = NULL;
            this->northEast = NULL;
            this->southWest = NULL;
            this->southEast = NULL;
            this->data = NULL;
            this->ldim = 0;
            this->m = 0;
            this->n = 0;
            memset(this->dimSub, 0, sizeof(int) * 4);
            
            // validate metadata
            if (isValidated == false)
            {
                bool isValid = this->ValidateMetadata(meta, metaLen, dataLen);
                if (isValid == false)
                {
                    mexErrMsgIdAndTxt("MATLAB:hmat:HMat:badMeta",
                        "Supplied metadata is invalid; a default object will be returned.");
                    return;
                }
            }
            
            // determine from first row whether this is a dense matrix or not
            if (meta[NUMMETA*metaIdx] == 0)
            {
                // get matrix dimensions
                this->m = meta[NUMMETA*metaIdx + 1];
                this->n = meta[NUMMETA*metaIdx + 2];
                this->data = (T*)mxMalloc(sizeof(T) * this->m * this->n);
                this->ldim = this->m;
                memcpy(this->data, dataIn + meta[NUMMETA*metaIdx + 5], sizeof(T) * this->m * this->n);
                dimSub[0] = this->m;
                dimSub[1] = this->n;
            }
            else
            {
                // get dimensions
                memcpy(this->dimSub, meta + NUMMETA*metaIdx + 1, sizeof(int) * 4);
                this->m = this->dimSub[0] + this->dimSub[2];
                this->n = this->dimSub[1] + this->dimSub[3];
                
                // build low-rank components
                int neRank = meta[NUMMETA*metaIdx + 9];
                int offset = neRank * this->dimSub[0];
                this->northEast = new LowRankMat<T>(dataIn + meta[NUMMETA*metaIdx + 5], this->dimSub[0], this->dimSub[0],
                                                    dataIn + meta[NUMMETA*metaIdx + 5] + offset, this->dimSub[3], this->dimSub[3],
                                                    neRank);
                if (neRank > imin(this->dimSub[0], this->dimSub[3]))
                    (this->northEast)->CompressVectors();
                int swRank = meta[NUMMETA*metaIdx + 10];
                offset = swRank * this->dimSub[2];
                this->southWest = new LowRankMat<T>(dataIn + meta[NUMMETA*metaIdx + 6], this->dimSub[2], this->dimSub[2],
                                                    dataIn + meta[NUMMETA*metaIdx + 6] + offset, this->dimSub[1], this->dimSub[1],
                                                    swRank);
                if (swRank > imin(this->dimSub[1], this->dimSub[2]))
                    (this->southWest)->CompressVectors();
                
                // build hierarchical components
                this->northWest = new HMat<T>(dataIn, meta, dataLen, metaLen, meta[NUMMETA*metaIdx + 7], true);
                this->southEast = new HMat<T>(dataIn, meta, dataLen, metaLen, meta[NUMMETA*metaIdx + 8], true);
            }
        }
        
        // returns matrix dimensions
        void Dims(int *a) const
        {
            if (a == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:hmat:HMat:Dims:badInput",
                                  "Supplied vector to put dimensions into is NULL.");
            }
            else
            {
                a[0] = this->m;
                a[1] = this->n;
            }
            return;
        }
        
        // returns breakdown of matrix dimensions
        void DimsSub(int *a) const
        {
            if (a == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:hmat:HMat:DimsSub:badInput",
                                  "Supplied vector to put dimensions into is NULL.");
            }
            else
                memcpy(a, this->dimSub, sizeof(int) * 4);
            return;
        }
        
        // returns whether matrix is actually dense
        bool IsDense() const
        {
            return (this->data != NULL);
        }
        
        // assignment operator
        HMat<T>& operator=(const HMat<T> rhs)
        {
            rhs.Swap(*this);
            return *this;
        }
        
        /************* operator definitions *************/
        
        // add and assign
        HMat<T>& operator += (const HMat<T>);
        
        // subtract and assign
        HMat<T>& operator -= (const HMat<T>);
        
        // add two hierarchical matrices
        HMat<T> operator + (const HMat<T>) const ;
        
        // subtract two hierarchical matrices
        HMat<T> operator - (const HMat<T>) const ;
        
        /************* specialized addition *************/
        
        // add a low-rank matrix to the hierarchical matrix
        void AddAssignLowRank(LowRankMat<T>*,int,int);
        
        // subtract a low-rank matrix from the hierarchical matrix
        void SubAssignLowRank(LowRankMat<T>*,int,int);
        
        // add a low-rank matrix to the hierarchical matrix
        HMat<T>* AddLowRank(LowRankMat<T>*,int,int);
        
        // subtract a low-rank matrix from the hierarchical matrix
        HMat<T>* SubLowRank(LowRankMat<T>*,int,int);
        
        /************* specialized multiplication *************/
        
        // right-multiply by a dense matrix
        T* RMultDense(T*,int,int,int&) const ;
        
        // left-multiply by a dense matrix
        T* LMultDense(T*,int,int,int&) const ;
        
        // right-multiply by a low-rank matrix
        LowRankMat<T>* RMultLowRank(LowRankMat<T>*, int,int,int) const ;
        
        // left-multiply by a low-rank matrix
        LowRankMat<T>* LMultLowRank(LowRankMat<T>*, int,int,int) const ;
        
        // right-multiply by a hierarchical matrix and assign
        void RMultAssignHMat(HMat<T>*);
        
        // left-multiply by a hierarchical matrix and assign
        void LMultAssignHMat(HMat<T>*);
        
        // right-multiply by a hierarchical matrix
        HMat<T>* RMultHMat(HMat<T>*);
        
        // left-multiply by a hierarchical matrix
        HMat<T>* LMultHMat(HMat<T>*);
        
        /************* other *************/
        
        // apply the inverse to the left of an input
        void ApplyInverseLeft(T*, const int, const int) const;
        
        // compute the inverse
        HMat<T>* Invert() const;
        
        // make a dense version
        void MakeDense(T**, int&) const;
        
        // returns information about the matrix
        void About() const
        {
            mexPrintf("--------------------------------------\n");
            if ((this->northWest == NULL) && (this->northEast == NULL) &&
                (this->southWest == NULL) && (this->southEast == NULL))
            {
                mexPrintf("\t Dense matrix, %d x %d \t\n", this->m, this->n);
                mexPrintf("--------------------------------------\n");
            }
            else
            {
                
                mexPrintf("\t Hierarchical matrix \t\n");
                int dimSubs[2];
                if (this->northWest == NULL)
                    mexPrintf("\t Northwest:  NULL\n");
                else
                {
                    (this->northWest)->Dims(dimSubs);
                    mexPrintf("\t Northwest: %d x %d\n", dimSubs[0], dimSubs[1]);
                }
                if (this->northEast == NULL)
                    mexPrintf("\t Northeast:  NULL\n");
                else
                {
                    (this->northEast)->Dims(dimSubs);
                    mexPrintf("\t Northeast: %d x %d (rank %d)\n", dimSubs[0], dimSubs[1], (this->northEast)->Rank());
                }
                if (this->southWest == NULL)
                    mexPrintf("\t Southwest:  NULL\n");
                else
                {
                    (this->southWest)->Dims(dimSubs);
                    mexPrintf("\t Southwest: %d x %d (rank %d)\n", dimSubs[0], dimSubs[1], (this->southWest)->Rank());
                }
                if (this->southEast == NULL)
                    mexPrintf("\t Southeast:  NULL\n");
                else
                {
                    (this->southEast)->Dims(dimSubs);
                    mexPrintf("\t Southeast: %d x %d\n", dimSubs[0], dimSubs[1]);
                }
                mexPrintf("--------------------------------------\n");
                if (this->northWest != NULL)
                    (this->northWest)->About();
                if (this->southEast != NULL)
                    (this->southEast)->About();
            }
        }
        
        // returns the total amount of storage required for this matrix
        int Storage(int &nsub) const
        {
            int storage = 0;
            if (this->IsDense())
            {
                storage = this->m * this->n;
                nsub++;
            }
            else
            {
                nsub++;
                storage = (this->northWest)->Storage(nsub) + (this->southEast)->Storage(nsub) +
                          (this->southWest)->Storage() + (this->northEast)->Storage();
            }
            return storage;
        }
        
        // returns storage in hierarhical form
        void GenerateOutput(T**, int**, int&, int&) const;
    
    private:
        
        // quad-tree children
        HMat<T> *northWest, *southEast;
        LowRankMat<T> *northEast, *southWest;
        
        // should only be non-NULL if this is the finest level of the tree
        T* data;
        int ldim;
        
        // matrix dimensions
        int m, n;
        int dimSub[4];
        
        // swap data elements
        void Swap(HMat &s)
        {
            std::swap(this->northWest, s.northWest);
            std::swap(this->southEast, s.southEast);
            std::swap(this->northEast, s.northEast);
            std::swap(this->southWest, s.southWest);
            std::swap(this->data, s.data);
            std::swap(this->ldim, s.ldim);
            std::swap(this->m, s.m);
            std::swap(this->n, s.n);
            std::swap(this->dimSub, s.dimSub);
        }
        
        // places storage in hierarchical form
        void PlaceOutput(T*, int*, int&, int&) const;
        
        // validates metadata
        bool ValidateMetadata(int*, int, int) const;
        
};

// compare the dimensions of two hierarchical matrices
template <class T> bool DimMatch(HMat<T>* const rhs1, HMat<T>* const rhs2)
{
    // input validation
    if ((rhs1 == NULL) || (rhs2 == NULL))
        mexErrMsgIdAndTxt("MATLAB:hmat:DimMatch:invalidInput",
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

// addition-assignment of hierarhical matrices
template <class T> HMat<T>& HMat<T>::operator += (const HMat<T> rhs)
{
    // first check - are dimensions equal
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:PlusEqual:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // next check if either side (or both) is actually dense
    if (this->IsDense() || rhs.IsDense())
    {
        // both matrices are dense - simple add
        if (this->IsDense() && rhs.IsDense())
        {
            for (int j = 0; j < rhs.n; j++)
                for (int i = 0; i < rhs.m; i++)
                    this->data[i + this->ldim*j] += rhs.data[i + rhs.ldim*j];
        }
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:hmat:HMat:PlusEqual:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the hierarhical structures are different
        (*(this->northWest)) += (*(rhs.northWest));
        (*(this->southEast)) += (*(rhs.southEast));
        (*(this->northEast)) += (*(rhs.northEast));
        (*(this->southWest)) += (*(rhs.southWest));
    }
}

// subtraction-assignment of hierarhical matrices
template <class T> HMat<T>& HMat<T>::operator -= (const HMat<T> rhs)
{
    // first check - are dimensions equal
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:MinusEqual:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // next check if either side (or both) is actually dense
    if (this->IsDense() || rhs.IsDense())
    {
        // both matrices are dense - simple add
        if (this->IsDense() && rhs.IsDense())
        {
            for (int j = 0; j < rhs.n; j++)
                for (int i = 0; i < rhs.m; i++)
                    this->data[i + this->ldim*j] -= rhs.data[i + rhs.ldim*j];
        }
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:hmat:HMat:MinusEqual:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the hierarhical structures are different
        (*(this->northWest)) -= (*(rhs.northWest));
        (*(this->southEast)) -= (*(rhs.southEast));
        (*(this->northEast)) -= (*(rhs.northEast));
        (*(this->southWest)) -= (*(rhs.southWest));
    }
}

// add two hierarchical matrices to form a new instance
template <class T> HMat<T> HMat<T>::operator + (const HMat<T> rhs) const
{
    // first check - are dimensions equal
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:Plus:dimMismatch",
                          "Matrix dimensions are not equal.");
    
    // initialize solution
    HMat<T> lhs(rhs);
    
    // next check if either side (or both) is actually dense
    if (this->IsDense() || rhs.IsDense())
    {
        // both matrices are dense - simple add
        if (this->IsDense() && rhs.IsDense())
        {
            for (int j = 0; j < rhs.n; j++)
                for (int i = 0; i < rhs.m; i++)
                    lhs.data[i + rhs.m*j] += this->data[i + this->ldim*j];
        }
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:hmat:HMat:Plus:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the hierarhical structures are different
        (*(lhs.northWest)) += (*(this->northWest));
        (*(lhs.southEast)) += (*(this->southEast));
        (*(lhs.northEast)) += (*(this->northEast));
        (*(lhs.southWest)) += (*(this->southWest));
    }
    return lhs;
}

// add two hierarchical matrices to form a new instance
template <class T> HMat<T> HMat<T>::operator - (const HMat<T> rhs) const
{
    // first check - are dimensions equal
    int dimRhs[2];
    rhs.Dims(dimRhs);
    
    // check dimension match
    if ((dimRhs[0] != this->m) || (dimRhs[1] != this->n))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:Minus:dimMismatch",
                          "Matrix dimensions are not equal.");
    
    // initialize solution
    HMat<T> lhs(*this);
    
    // next check if either side (or both) is actually dense
    if (this->IsDense() || rhs.IsDense())
    {
        // both matrices are dense - simple add
        if (this->IsDense() && rhs.IsDense())
        {
            for (int j = 0; j < lhs.n; j++)
                for (int i = 0; i < lhs.m; i++)
                    lhs.data[i + lhs.ldim*j] -= rhs.data[i + rhs.ldim*j];
        }
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:hmat:HMat:Minus:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the hierarhical structures are different
        (*(lhs.northWest)) -= (*(rhs.northWest));
        (*(lhs.southEast)) -= (*(rhs.southEast));
        (*(lhs.northEast)) -= (*(rhs.northEast));
        (*(lhs.southWest)) -= (*(rhs.southWest));
    }
    return lhs;
}

/************* specialized addition *************/

//add a low-rank matrix to the hierarchical matrix
template <class T> void HMat<T>::AddAssignLowRank(LowRankMat<T>* rhs, const int rI=0, const int cI=0)
{
    // get low-rank dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);

    // check indices
    if ((rI < 0) || (rI >= dimRhs[0]) || (cI < 0) || (cI >= dimRhs[1]))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:AddAssignLowRank:badSubmat",
                          "Specified indices for submatrix are out of possible range.");
    
    // check against this matrix's dimensions
    if (((dimRhs[0] - rI) < this->m) || ((dimRhs[1] - cI) < this->n))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:AddAssignLowRank:dimMismatch",
                          "Matrix dimensions do not agree.");
        
    // dense addition
    if (this->IsDense())
    {
        // convert low-rank to dense
        int rhsLdim;
        T* temp = NULL;
        rhs->MakeDense(&temp, rhsLdim, rI, cI);
        // add to dense version of this matrix
        for (int j = 0; j < this->n; j++)
        {
            for (int i = 0; i < this->m; i++)
                this->data[i + j*this->ldim] += temp[i + j*rhsLdim];
        }
        // free
        mxFree(temp);
    }
    // hierarchical addition
    else
    {
        // get dimensions of subcomponents
        int dimSubs[2];
        (this->northWest)->Dims(dimSubs);
        
        // add to hierarchical components
        (this->northWest)->AddAssignLowRank(rhs, rI, cI);
        (this->southEast)->AddAssignLowRank(rhs, rI + dimSubs[0], cI + dimSubs[1]);
        
        // add to low-rank components
        (this->northEast)->AddAssignPart(rhs, rI, cI + dimSubs[1]);
        (this->southWest)->AddAssignPart(rhs, rI + dimSubs[0], cI);
    }
}

// subtract a low-rank matrix to the hierarchical matrix
template <class T> void HMat<T>::SubAssignLowRank(LowRankMat<T>* rhs, const int rI=0, const int cI=0)
{
    // get low-rank dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);

    // check indices
    if ((rI < 0) || (rI >= dimRhs[0]) || (cI < 0) || (cI >= dimRhs[1]))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:SubAssignLowRank:badSubmat",
                          "Specified indices for submatrix are out of possible range.");
    
    // check against this matrix's dimensions
    if (((dimRhs[0] - rI) < this->m) || ((dimRhs[1] - cI) < this->n))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:SubAssignLowRank:dimMismatch",
                          "Matrix dimensions do not agree.");
        
    // dense addition
    if (this->IsDense())
    {
        // convert low-rank to dense
        int rhsLdim;
        T* temp = NULL;
        rhs->MakeDense(&temp, rhsLdim, rI, cI);
        // add to dense version of this matrix
        for (int j = 0; j < this->n; j++)
        {
            for (int i = 0; i < this->m; i++)
                this->data[i + j*this->ldim] -= temp[i + j*rhsLdim];
        }
        // free
        mxFree(temp);
    }
    // hierarchical addition
    else
    {
        // get dimensions of subcomponents
        int dimSubs[2];
        (this->northWest)->Dims(dimSubs);
        
        // add to hierarchical components
        (this->northWest)->SubAssignLowRank(rhs, rI, cI);
        (this->southEast)->SubAssignLowRank(rhs, rI + dimSubs[0], cI + dimSubs[1]);
        
        // add to low-rank components
        (this->northEast)->SubAssignPart(rhs, rI, cI + dimSubs[1]);
        (this->southWest)->SubAssignPart(rhs, rI + dimSubs[0], cI);
    }
}

// add a low-rank matrix to the hierarchical matrix
template <class T> HMat<T>* HMat<T>::AddLowRank(LowRankMat<T>* rhs, const int rI=0, const int cI=0)
{
    HMat<T>* lhs = new HMat<T>(*this);
    lhs->AddAssignLowRank(rhs, rI, cI);
    return lhs;
}

// subtract a low-rank matrix from the hierarchical matrix
template <class T> HMat<T>* HMat<T>::SubLowRank(LowRankMat<T>* rhs, const int rI=0, const int cI=0)
{
    HMat<T>* lhs = new HMat<T>(*this);
    lhs->SubAssignLowRank(rhs, rI, cI);
    return lhs;
}

/************* specialized multiplication *************/

// right-multiply by a dense matrix
template <class T> T* HMat<T>::RMultDense(T* rhs, const int nCols, const int ldimIn, int& ldimOut) const 
{
    // check input column
    if (rhs == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultDense:nullInput",
                          "Supplied input data is NULL.");
    }
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultDense:badNumCols",
                          "Specified number of columns is invalid.");
    }  
    if (ldimIn < this->n)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultDense:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    // perform multiplication
    T* ans = (T*)mxMalloc(sizeof(T) * this->m * nCols);
    ldimOut = this->m;
    memset(ans, 0, sizeof(T) * this->m * nCols);
    if (this->IsDense() == true)
    {
        char transA = 'N', transB = 'N';
        // perform matrix multiplication
        int numCols = nCols;
        int ldl = this->ldim;
        int ldr = ldimIn;
        int outerDim = this->m, innerDim = this->n;
        
        if (sizeof(T) == sizeof(cmpx))
        {
            // matrix multiply
            cmpx alpha = 1.0, beta = 0.0;
            zgemm(&transA, &transB, &outerDim, &numCols, &innerDim,
                  CCAST(&alpha), CCAST(this->data), &ldl, CCAST(rhs), &ldr,
                  CCAST(&beta), CCAST(ans), &outerDim);
        }
        else
        {
            if (sizeof(T) == sizeof(double))
            {
                // matrix multiply
                double alpha = 1.0, beta = 0.0;
                dgemm(&transA, &transB, &outerDim, &numCols, &innerDim,
                      &alpha, (double*)(this->data), &ldl, (double*)(rhs), &ldr,
                      &beta, (double*)ans, &outerDim);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:RMultDense:invalidDataType",
                                   "Cannot multiply because data type is not double or cmpx.");
                mxFree(ans);
                return NULL;
            }
        }
        return ans;
    }
    else
    {
        // compute northern half
        int tmpldo = 0;
        T *temp = (this->northWest)->RMultDense(rhs, nCols, ldimIn, tmpldo);
        for (int i = 0; i < nCols; i++)
            memcpy(ans + i*this->m, temp + i*tmpldo, sizeof(T) * dimSub[0]);
        mxFree(temp);
        temp = (this->northEast)->RMultDense(rhs + dimSub[1], nCols, ldimIn, tmpldo);
        for (int i = 0; i < nCols; i++)
        {
            for (int j = 0; j < dimSub[0]; j++)
                ans[j + i*this->m] += temp[j + i*tmpldo];
        }
        mxFree(temp);
        
        // compute southern half
        temp = (this->southEast)->RMultDense(rhs + dimSub[1], nCols, ldimIn, tmpldo);
        for (int i = 0; i < nCols; i++)
            memcpy(ans + i*this->m + dimSub[0], temp + i*tmpldo, sizeof(T) * dimSub[2]);
        mxFree(temp);
            
        temp = (this->southWest)->RMultDense(rhs, nCols, ldimIn, tmpldo);
        for (int i = 0; i < nCols; i++)
        {
            for (int j = 0; j < dimSub[1]; j++)
                ans[dimSub[0] + j + i*this->m] += temp[j + i*tmpldo];
        }
        mxFree(temp);
    }
    
    return ans;
}

// right-multiply by a dense matrix
template <class T> T* HMat<T>::LMultDense(T* rhs, const int nRows, const int ldimIn, int& ldimOut) const 
{
    // check input column
    if (rhs == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultDense:nullInput",
                          "Supplied input data is NULL.");
    }
    if (nRows < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultDense:badNumCols",
                          "Specified number of rows is invalid.");
    }  
    if (ldimIn < nRows)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultDense:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    // perform multiplication
    T* ans = (T*)mxMalloc(sizeof(T) * nRows * this->n);
    ldimOut = nRows;
    memset(ans, 0, sizeof(T) * nRows * this->n);
    if (this->IsDense() == true)
    {
        char transA = 'N', transB = 'N';
        // perform matrix multiplication
        int numRows = nRows;
        int ldl = ldimIn;
        int ldr = this->ldim;
        int outerDim = this->n, innerDim = this->m;
        
        if (sizeof(T) == sizeof(cmpx))
        {
            // matrix multiply
            cmpx alpha = 1.0, beta = 0.0;
            zgemm(&transA, &transB, &numRows, &outerDim, &innerDim,
                  CCAST(&alpha), CCAST(rhs), &ldl, CCAST(this->data), &ldr,
                  CCAST(&beta), CCAST(ans), &numRows);
        }
        else
        {
            if (sizeof(T) == sizeof(double))
            {
                // matrix multiply
                double alpha = 1.0, beta = 0.0;
                dgemm(&transA, &transB, &numRows, &outerDim, &innerDim,
                      &alpha, (double*)(rhs), &ldl,(double*)(this->data), &ldr, 
                      &beta, (double*)ans, &numRows);
            }
            else
            {
                mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:LMultDense:invalidDataType",
                                   "Cannot multiply because data type is not double or cmpx.");
                mxFree(ans);
                return NULL;
            }
        }
        return ans;
    }
    else
    {   
        // compute western half
        int tmpldo = 0;
        T *temp = (this->northWest)->LMultDense(rhs, nRows, ldimIn, tmpldo);
        for (int i = 0; i < dimSub[1]; i++)
            memcpy(ans + i*nRows, temp + i*tmpldo, sizeof(T) * nRows);
        mxFree(temp);
        
        temp = (this->southWest)->LMultDense(rhs + ldimIn * dimSub[0], nRows, ldimIn, tmpldo);
        for (int i = 0; i < dimSub[1]; i++)
        {
            for (int j = 0; j < nRows; j++)
                ans[j + i*nRows] += temp[j + i*tmpldo];
        }
        mxFree(temp);
        
        // compute eastern half
        temp = (this->southEast)->LMultDense(rhs + ldimIn * dimSub[0], nRows, ldimIn, tmpldo);
        for (int i = 0; i < dimSub[3]; i++)
            memcpy(ans + (dimSub[1] + i)*nRows, temp + i*tmpldo, sizeof(T) * nRows);
        mxFree(temp);
            
        temp = (this->northEast)->LMultDense(rhs, nRows, ldimIn, tmpldo);
        for (int i = 0; i < dimSub[3]; i++)
        {
            for (int j = 0; j < nRows; j++)
                ans[j + (dimSub[1] + i)*nRows] += temp[j + i*tmpldo];
        }
        mxFree(temp);
    }
    
    return ans;
}

// right-multiply by a low-rank matrix
template <class T> LowRankMat<T>* HMat<T>::RMultLowRank(LowRankMat<T>* rhs, const int rI, const int cI, const int nCols) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check dimensions
    if ((rI < 0) || ((dimRhs[0] - rI) < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultLowRank:badInput",
                          "Supplied row index is out of range.");
    }
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultLowRank:badInput",
                          "Supplied number of columns is invalid.");
    }
    if ((cI < 0) || ((dimRhs[1] - cI) < nCols))
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultLowRank:badInput",
                          "Supplied row index is out of range.");
    }
    
    // allocate for the matrix
    int rankVal = rhs->Rank();
    LowRankMat<T>* lhs = NULL;
    
    // check if the matrix is dense
    if (this->IsDense() == true)
    {
        // start by creating a submatrix from the input matrix
        lhs = rhs->Submat(rI, this->n, cI, nCols);
        
        // then multiply by the matrix
        lhs->LMultAssignPart(this->data, this->m, this->ldim);        
    }
    else
    {
        // get the left vectors from the rhs
        T* temp = (T*)mxMalloc(sizeof(T) * this->n * rankVal);
        rhs->CopyLeftData(temp, this->n, rI, this->n);
        
        // get the new left vectors
        int ldout = 0;
        T* newLeft = this->RMultDense(temp, rankVal, this->n, ldout);
        mxFree(temp);
        
        temp = (T*)mxMalloc(sizeof(T) * nCols * rankVal);
        rhs->CopyRightData(temp, nCols, cI, nCols);
        lhs = new LowRankMat<T>(newLeft, ldout, this->m, temp, nCols, nCols, rankVal);
        mxFree(temp);
    }
    return lhs;
}

// right-multiply by a low-rank matrix
template <class T> LowRankMat<T>* HMat<T>::LMultLowRank(LowRankMat<T>* rhs, const int rI, const int cI, const int nRows) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check dimensions
    if (nRows < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultLowRank:badInput",
                          "Supplied number of rows is invalid.");
    }
    if ((rI < 0) || ((dimRhs[0] - rI) < nRows))
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultLowRank:badInput",
                          "Supplied row index is out of range.");
    }
    
    if ((cI < 0) || ((dimRhs[1] - cI) < this->m))
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultLowRank:badInput",
                          "Supplied row index is out of range.");
    }
    
    // allocate for the matrix
    int rankVal = rhs->Rank();
    LowRankMat<T>* lhs = NULL;
    
    // check if the matrix is dense
    if (this->IsDense() == true)
    {
        // start by creating a submatrix from the input matrix
        lhs = rhs->Submat(rI, nRows, cI, this->m);
        
        // then multiply by the matrix
        lhs->RMultAssignPart(this->data, this->n, this->ldim);        
    }
    else
    {
        // get the right vectors from the rhs
        T* temp = (T*)mxMalloc(sizeof(T) * this->m * rankVal);
        rhs->CopyRightData(temp, this->m, cI, this->m);
        
        // transpose the right vectors
        int ldtemp = this->m;
        TransposeInPlace(&temp, this->m, rankVal, ldtemp);
        
        // get the new right vectors
        int ldout = 0;
        T* newRight = this->LMultDense(temp, rankVal, ldtemp, ldout);
        mxFree(temp);
        
        // transpose the new right vectors
        TransposeInPlace(&newRight, rankVal, this->n, ldout);
        temp = (T*)mxMalloc(sizeof(T) * nRows * rankVal);
        rhs->CopyLeftData(temp, nRows, rI, nRows);
        
        // create new matrix
        lhs = new LowRankMat<T>(temp, nRows, nRows, newRight, ldout, this->n, rankVal);

        // cleanup
        mxFree(temp);
        mxFree(newRight);
    }
    return lhs;
}

// right-multiply by a hierarchical matrix and assign
template <class T> void HMat<T>::RMultAssignHMat(HMat<T>* rhs)
{
    // input validation
    if (rhs == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultAssignHMat:badInput",
                          "Input hierarchical matrix is NULL.");
    }
    if (this->n != rhs->m)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultAssignHMat:dimMismatch",
                          "Inner dimensions of the matrices do not agree.");
    }
    
    // check if either matrix (or both) is dense
    if (this->IsDense() || rhs->IsDense())
    {
        // both matrices are dense - dense multiply
        if (this->IsDense() && rhs->IsDense())
        {
            int ldata = 0;
            T* lhsData = this->RMultDense(rhs->data, rhs->n, rhs->ldim, ldata);
            mxFree(this->data);
            this->data = lhsData;
            this->ldim = ldata;
            this->n = rhs->n;
        }
        // convert one of the matrices to dense
        else
        {
            // this matrix is dense, rhs is hierarchical
            if (this->IsDense())
            {
                int ldata = 0;
                T* lhsData = rhs->LMultDense(this->data, this->m, this->ldim, ldata);
                mxFree(this->data);
                this->data = lhsData;
                this->ldim = ldata;
                this->n = rhs->n;
            }
            // this matrix is hierarchical, rhs is dense
            if (rhs->IsDense())
            {
                // convert this matrix to a dense matrix
                this->MakeDense(&(this->data), this->ldim);
                if (this->northWest != NULL)
                {
                    delete this->northWest;
                    this->northWest = NULL;
                }
                if (this->northEast != NULL)
                {
                    delete this->northEast;
                    this->northEast = NULL;
                }
                if (this->southWest != NULL)
                {
                    delete this->southWest;
                    this->southWest = NULL;
                }
                if (this->southEast != NULL)
                {
                    delete this->southEast;
                    this->southEast = NULL;
                }
                if (this->data != NULL)
                {
                    mxFree(this->data);
                    this->data = NULL;
                }
                memset(this->dimSub, 0, sizeof(int) * 4);
                
                // do a standard dense-times-dense product
                int ldata = 0;
                T* lhsData = this->RMultDense(rhs->data, rhs->n, rhs->ldim, ldata);
                mxFree(this->data);
                this->data = lhsData;
                this->ldim = ldata;
                this->n = rhs->n;
            }
        }
    }
    else
    {
        // check dimensions for compatibility
        if ((this->dimSub[1] != rhs->dimSub[0]) || (this->dimSub[3] != rhs->dimSub[2]))
        {
            mexErrMsgIdAndTxt("MATLAB:hmat:HMat:RMultAssignHMat:dimMismatch",
                              "Hierarhical structures of the matrices are not compatible for right multiplication.");
        }
        
        /* -------- north components -------- */
        // (1) compute hierarhical portion of northwest
        HMat<T>* newNorthWest = new HMat<T>(*(this->northWest));
        newNorthWest->RMultAssignHMat(rhs->northWest);
        // (2) compute low-rank portion of northwest
        LowRankMat<T>* temp = (this->northEast)->RMultPart(rhs->southWest, 0, 0, rhs->dimSub[1]);
        newNorthWest->AddAssignLowRank(temp, 0, 0);
        delete temp;
        // (3) compute first low-rank portion of northeast
        temp = (this->northWest)->RMultLowRank(rhs->northEast, 0, 0, rhs->dimSub[3]);
        // (4) finished with old northwest, delete and assign new northwest
        delete this->northWest;
        this->northWest = newNorthWest;
        // (5) compute second low-rank portion of northeast
        LowRankMat<T>* temp2 = (rhs->southEast)->LMultLowRank(this->northEast, 0, 0, this->dimSub[0]);
        // (6) finished with old northeast, delete and assign new northeast
        delete this->northEast;
        this->northEast = new LowRankMat<T>(*temp2);
        delete temp2;
        (this->northEast)->AddAssignPart(temp, 0, 0);
        delete temp;
        
        /* -------- south components -------- */
        // (1) compute hierarhical portion of southeast
        HMat<T>* newSouthEast = new HMat<T>(*(this->southEast));
        newSouthEast->RMultAssignHMat(rhs->southEast);
        // (2) compute low-rank portion of southeast
        temp = (this->southWest)->RMultPart(rhs->northEast, 0, 0, rhs->dimSub[3]);
        newSouthEast->AddAssignLowRank(temp, 0, 0);
        delete temp;
        // (3) compute first low-rank portion of southwest
        temp = (this->southEast)->RMultLowRank(rhs->southWest, 0, 0, rhs->dimSub[1]);
        // (4) finished with old southeast, delete and assign new southeast
        delete this->southEast;
        this->southEast = newSouthEast;
        // (5) compute second low-rank portion of southwest
        temp2 = (rhs->northWest)->LMultLowRank(this->southWest, 0, 0, this->dimSub[2]);
        // (6) finished with old southwest, delete and assign new southwest
        delete this->southWest;
        this->southWest = new LowRankMat<T>(*temp2);
        delete temp2;
        (this->southWest)->AddAssignPart(temp, 0, 0);
        delete temp;
        
        /* -------- update subdimensions -------- */
        this->dimSub[1] = rhs->dimSub[1];
        this->dimSub[3] = rhs->dimSub[3];
    }
}

// left-multiply by a hierarchical matrix
template <class T> void HMat<T>::LMultAssignHMat(HMat<T>* rhs)
{
    // input validation
    if (rhs == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultAssignHMat:badInput",
                          "Input hierarchical matrix is NULL.");
    }
    if (this->m != rhs->n)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultAssignHMat:dimMismatch",
                          "Inner dimensions of the matrices do not agree.");
    }
    
    // check if either matrix (or both) is dense
    if (this->IsDense() || rhs->IsDense())
    {
        // both matrices are dense - dense multiply
        if (this->IsDense() && rhs->IsDense())
        {
            int ldata = 0;
            T* lhsData = this->LMultDense(rhs->data, rhs->m, rhs->ldim, ldata);
            mxFree(this->data);
            this->data = lhsData;
            this->ldim = ldata;
            this->m = rhs->m;
        }
        // convert one of the matrices to dense
        else
        {
            // this matrix is dense, rhs is hierarchical
            if (this->IsDense())
            {
                int ldata = 0;
                T* lhsData = rhs->RMultDense(this->data, this->n, this->ldim, ldata);
                mxFree(this->data);
                this->data = lhsData;
                this->ldim = ldata;
                this->m = rhs->m;
            }
            // this matrix is hierarchical, rhs is dense
            if (rhs->IsDense())
            {
                // convert this matrix to a dense matrix
                this->MakeDense(&(this->data), this->ldim);
                if (this->northWest != NULL)
                {
                    delete this->northWest;
                    this->northWest = NULL;
                }
                if (this->northEast != NULL)
                {
                    delete this->northEast;
                    this->northEast = NULL;
                }
                if (this->southWest != NULL)
                {
                    delete this->southWest;
                    this->southWest = NULL;
                }
                if (this->southEast != NULL)
                {
                    delete this->southEast;
                    this->southEast = NULL;
                }
                if (this->data != NULL)
                {
                    mxFree(this->data);
                    this->data = NULL;
                }
                memset(this->dimSub, 0, sizeof(int) * 4);
                
                // do a standard dense-times-dense product
                int ldata = 0;
                T* lhsData = this->LMultDense(rhs->data, rhs->m, rhs->ldim, ldata);
                mxFree(this->data);
                this->data = lhsData;
                this->ldim = ldata;
                this->m = rhs->m;
            }
        }
    }
    else
    {
        // check dimensions for compatibility
        if ((this->dimSub[0] != rhs->dimSub[1]) || (this->dimSub[2] != rhs->dimSub[3]))
        {
            mexErrMsgIdAndTxt("MATLAB:hmat:HMat:LMultAssignHMat:dimMismatch",
                              "Hierarhical structures of the matrices are not compatible for right multiplication.");
        }
        
        /* -------- west components -------- */
        // (1) compute hierarhical portion of northwest
        HMat<T>* newNorthWest = new HMat<T>(*(this->northWest));
        newNorthWest->LMultAssignHMat(rhs->northWest);
        // (2) compute low-rank portion of northwest
        LowRankMat<T>* temp = (this->southWest)->LMultPart(rhs->northEast, 0, 0, rhs->dimSub[0]);
        newNorthWest->AddAssignLowRank(temp, 0, 0);
        delete temp;
        // (3) compute first low-rank portion of southwest
        temp = (this->northWest)->LMultLowRank(rhs->southWest, 0, 0, rhs->dimSub[2]);
        // (4) finished with old northwest, delete and assign new northwest
        delete this->northWest;
        this->northWest = newNorthWest;
        // (5) compute second low-rank portion of southwest
        LowRankMat<T>* temp2 = (rhs->southEast)->RMultLowRank(this->southWest, 0, 0, this->dimSub[1]);
        // (6) finished with old southwest, delete and assign new southwest
        delete this->southWest;
        this->southWest = new LowRankMat<T>(*temp2);
        delete temp2;
        (this->southWest)->AddAssignPart(temp, 0, 0);
        delete temp;
        
        /* -------- east components -------- */
        // (1) compute hierarhical portion of southeast
        HMat<T>* newSouthEast = new HMat<T>(*(this->southEast));
        newSouthEast->LMultAssignHMat(rhs->southEast);
        // (2) compute low-rank portion of southeast
        temp = (this->northEast)->LMultPart(rhs->southWest, 0, 0, rhs->dimSub[2]);
        newSouthEast->AddAssignLowRank(temp, 0, 0);
        delete temp;
        // (3) compute first low-rank portion of northeast
        temp = (this->southEast)->LMultLowRank(rhs->northEast, 0, 0, rhs->dimSub[0]);
        // (4) finished with old southeast, delete and assign new southeast
        delete this->southEast;
        this->southEast = newSouthEast;
        // (5) compute second low-rank portion of northeast
        temp2 = (rhs->northWest)->RMultLowRank(this->northEast, 0, 0, this->dimSub[3]);
        // (6) finished with old northeast, delete and assign new northeast
        delete this->northEast;
        this->northEast = new LowRankMat<T>(*temp2);
        delete temp2;
        (this->northEast)->AddAssignPart(temp, 0, 0);
        delete temp;
        
        /* -------- update subdimensions -------- */
        this->dimSub[0] = rhs->dimSub[0];
        this->dimSub[2] = rhs->dimSub[2];
    }
}

// right-multiply by a hierarchical matrix
template <class T> HMat<T>* HMat<T>::RMultHMat(HMat<T>* rhs)
{
    HMat<T>* lhs = new HMat<T>(*this);
    lhs->RMultAssign(rhs);
    return lhs;
}

// left-multiply by a hierarchical matrix
template <class T> HMat<T>* HMat<T>::LMultHMat(HMat<T>* rhs)
{
    HMat<T>* lhs = new HMat<T>(*this);
    lhs->LMultAssign(rhs);
    return lhs;
}

/************* other *************/

// apply the inverse to the left of an input
template <class T> void HMat<T>::ApplyInverseLeft(T* rhs, const int nCols, const int ldimIn) const
{
    // input validation
    if (rhs == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:ApplyInverseLeft:badInput",
                          "Input data matrix is NULL.");
    }
    if (ldimIn < this->n)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:ApplyInverseLeft:badInput",
                          "Leading dimension of input data is invalid.");
    }
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:ApplyInverseLeft:badInput",
                          "Input data must have at least one column");
    }
    // input validation
    if (this->m != this->n)
    {
        mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:Invert:singular",
                           "Matrix is singular and cannot be inverted.");
        return NULL;
    }
    if ((this->m == 0) || (this->n == 0))
    {
        mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:Invert:empty",
                           "Matrix is empty.");
        return NULL;
    }
    if ((this->data == NULL) && ((this->northWest == NULL) || (this->northEast == NULL) ||
                                 (this->southWest == NULL) || (this->southEast == NULL)))
    {
        mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:Invert:empty",
                           "Matrix is empty.");
        return NULL;
    }
    
    // check if this matrix is dense
    if (this->IsDense() == true)
    {
        // set order and size of the system to solve
        int order = this->n, nrhs = nCols;
        
        // copy system since _gesv will alter it
        T* sys = (T*)mxMalloc(sizeof(T) * order * order);
        for (int i = 0; i < order; i++)
            memcpy(sys + i*order, this->data + i*this->ldim, sizeof(T) * order);
            
        // allocate for pivot matrices
        int *ipiv = (int*)mxMalloc(sizeof(T) * order);
        int info = 0;
        int ld = ldimIn;
        
        // solve system with _gesv
        if (sizeof(T) == sizeof(double))
            dgesv(&order, &nrhs, (double*)sys, &order, ipiv, (double*)rhs, &ld, &info);
        else
        {
            if (sizeof(T) == sizeof(cmpx))
                zgesv(&order, &nrhs, CCAST(sys), &order, ipiv, CCAST(rhs), &ld, &info);
            else
            {
                mxFree(ipiv);
                mxFree(sys);
                mexErrMsgIdAndTxt("MATLAB:hmat:HMat:ApplyInverseLeft:invalidDataType",
                          "Cannot solve system because data type is not double or cmpx.");
            }
        }
        
        // free system copy and pivot indices
        mxFree(ipiv);
        mxFree(sys);
    }
    // otherwise
    else
    {
        // check which form of Schur recursion to use, optimizing op count
        int r1 = (this->southWest)->Rank(), r2 = (this->northEast)->Rank();
        if (r2 <= r1)
        {
            // (1) solve the northwest system for two input vectors
            int subOrder = dimSub[0], nCopy = (this->northEast)->Rank();
            T* temp1 = (T*)mxMalloc(sizeof(T) * subOrder * (nCopy + nCols));
            (this->northEast)->CopyLeftData(temp1, subOrder, 0, subOrder);
            for (int i = 0; i < nCols; i++)
                memcpy(temp1 + (i + nCopy)*subOrder, rhs + i*ldimIn, sizeof(T) * subOrder);
            (this->northWest)->ApplyInverseLeft(temp1, nCols + nCopy, subOrder);
            
            // (2) build the low-rank system
            T* temp2 = (T*)mxMalloc(sizeof(T) * nCopy * dimSub[3]);
            (this->northEast)->CopyRightData(temp2, dimSub[3], 0, dimSub[3]);
            LowRankMat<T> *ellCheck = new LowRankMat<T>(temp1, subOrder, subOrder,
                                                        temp2, dimSub[3], dimSub[3], nCopy);
            mxFree(temp2);
            
            // (3) build Schur complement
            LowRankMat<T> *tempMat = (this->southWest)->RMultPart(ellCheck, 0, 0, dimSub[3]);
            HMat<T> *S = new HMat<T>(*(this->southEast));
            S->SubAssignLowRank(tempMat);
            delete tempMat;
            
            // (4) build second component of solution
            int ldout = 0;
            T* temp3 = (this->southWest)->RMultDense(temp1 + subOrder*nCopy, nCols, subOrder, ldout);
            for (int j = 0; j < nCols; j++)
            {
                for (int i = 0; i < dimSub[2]; i++)
                    rhs[dimSub[0] + i + j*ldimIn] -= temp3[i + j*ldout];
            }
            mxFree(temp3);
            S->ApplyInverseLeft(rhs + dimSub[0], nCols, ldimIn);
            delete S;
            
            // (5) build first component of solution
            temp3 = ellCheck->RMultDense(rhs + dimSub[0], nCols, ldimIn, ldout);
            delete ellCheck;
            for (int j = 0; j < nCols; j++)
            {
                for (int i = 0; i < dimSub[0]; i++)
                    rhs[i + j*ldimIn] = temp1[i + (j + nCopy)*subOrder] - temp3[i + j*ldout];
            }
            mxFree(temp3);
            mxFree(temp1);
        }
        else
        {
            // (1) solve the southeast system for two input vectors
            int subOrder = dimSub[2], nCopy = (this->southWest)->Rank();
            T* temp1 = (T*)mxMalloc(sizeof(T) * subOrder * (nCopy + nCols));
            (this->southWest)->CopyLeftData(temp1, subOrder, 0, subOrder);
            for (int i = 0; i < nCols; i++)
                memcpy(temp1 + (i + nCopy)*subOrder, rhs + dimSub[0] + i*ldimIn, sizeof(T) * subOrder);
            (this->southEast)->ApplyInverseLeft(temp1, nCols + nCopy, subOrder);
            
            // (2) build the low-rank system
            T* temp2 = (T*)mxMalloc(sizeof(T) * nCopy * dimSub[1]);
            (this->southWest)->CopyRightData(temp2, dimSub[1], 0, dimSub[1]);
            LowRankMat<T> *ellCheck = new LowRankMat<T>(temp1, subOrder, subOrder,
                                                        temp2, dimSub[1], dimSub[1], nCopy);
            mxFree(temp2);
            
            // (3) build Schur complement
            LowRankMat<T> *tempMat = (this->northEast)->RMultPart(ellCheck, 0, 0, dimSub[1]);
            HMat<T> *S = new HMat<T>(*(this->northWest));
            S->SubAssignLowRank(tempMat);
            delete tempMat;
            
            // (4) build first component of solution
            int ldout = 0;
            T* temp3 = (this->northEast)->RMultDense(temp1 + subOrder*nCopy, nCols, subOrder, ldout);
            for (int j = 0; j < nCols; j++)
            {
                for (int i = 0; i < dimSub[2]; i++)
                    rhs[i + j*ldimIn] -= temp3[i + j*ldout];
            }
            mxFree(temp3);
            S->ApplyInverseLeft(rhs, nCols, ldimIn);
            delete S;
            
            // (5) build second component of solution
            temp3 = ellCheck->RMultDense(rhs, nCols, ldimIn, ldout);
            delete ellCheck;
            for (int j = 0; j < nCols; j++)
            {
                for (int i = 0; i < dimSub[0]; i++)
                    rhs[i + dimSub[0] + j*ldimIn] = temp1[i + (j + nCopy)*subOrder] - temp3[i + j*ldout];
            }
            mxFree(temp3);
            mxFree(temp1);
        }
    }
    
}

// compute the inverse
template <class T> HMat<T>* HMat<T>::Invert() const
{
    // input validation
    if (this->m != this->n)
    {
        mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:Invert:singular",
                           "Matrix is singular and cannot be inverted.");
        return NULL;
    }
    if ((this->m == 0) || (this->n == 0))
    {
        mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:Invert:empty",
                           "Matrix is empty.");
        return NULL;
    }
    if ((this->data == NULL) && ((this->northWest == NULL) || (this->northEast == NULL) ||
                                 (this->southWest == NULL) || (this->southEast == NULL)))
    {
        mexWarnMsgIdAndTxt("MATLAB:hmat:HMat:Invert:empty",
                           "Matrix is empty.");
        return NULL;
    }
    
    // dense
    if (this->IsDense() == true)
    {
        // create a copy of the data
        int order = this->n;
        T* invData = (T*)mxMalloc(sizeof(T) * order * order);
        for (int i = 0; i < order; i++)
            memcpy(invData + i*order, this->data + i*this->ldim, sizeof(T) * order);
        int *ipiv = (int*)mxMalloc(sizeof(int) * order), info;
        
        // invert
        if (sizeof(T) == sizeof(double))
        {
            // lu factorization
            dgetrf(&order, &order, (double*)invData, &order, ipiv, &info);
            
            // inversion
            double *work = NULL, workopt = 0;
            int lwork = -1;
            dgetri(&order, (double*)invData, &order, ipiv, &workopt, &lwork, &info);
            lwork = (int)workopt;
            work = (double*)mxMalloc(sizeof(double) * lwork);
            dgetri(&order, (double*)invData, &order, ipiv, work, &lwork, &info);
            mxFree(work);
        }
        else
        {
            // lu factorization
            zgetrf(&order, &order, CCAST(invData), &order, ipiv, &info);
            
            // inversion
            cmpx *work = NULL, workopt = 0;
            int lwork = -1;
            zgetri(&order, CCAST(invData), &order, ipiv, CCAST(&workopt), &lwork, &info);
            lwork = (int)real(workopt);
            work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
            zgetri(&order, CCAST(invData), &order, ipiv, CCAST(work), &lwork, &info);
            mxFree(work);
            
        }
        mxFree(ipiv);
        
        // create new hierarchical matrix structure
        HMat<T> *lhs = new HMat<T>(invData, order, order, order, order + 1);
        mxFree(invData);
        return lhs;
    }
    else
    {
        // initialize lhs structure
        HMat<T> *lhs = new HMat<T>();
        
        // set internal dimension variables
        lhs->m = this->m;
        lhs->n = this->n;
        memcpy(lhs->dimSub, this->dimSub, sizeof(int) * 4);
        
        // first invert northwest corner
        lhs->northWest = (this->northWest)->Invert();
        
        // compute special low rank matrices
        LowRankMat<T> *ellCheckOne = (lhs->northWest)->RMultLowRank(this->northEast, 0, 0, this->dimSub[3]);
        LowRankMat<T> *ellCheckTwo = (lhs->northWest)->LMultLowRank(this->southWest, 0, 0, this->dimSub[2]);
        
        // form the Schur complement
        HMat<T> *schur = new HMat<T>(*(this->southEast));
        LowRankMat<T> *temp = (this->southWest)->RMultPart(ellCheckOne, 0, 0, this->dimSub[3]);
        schur->SubAssignLowRank(temp);
        delete temp;
        
        // invert the Schur complement
        lhs->southEast = schur->Invert();
        delete schur;
        
        // compute the off-diagonal components
        lhs->southWest = (lhs->southEast)->RMultLowRank(ellCheckTwo, 0, 0, this->dimSub[1]);
        (lhs->southWest)->Negate(0, (lhs->southWest)->Rank());
        delete ellCheckTwo;
        lhs->northEast = (lhs->southEast)->LMultLowRank(ellCheckOne, 0, 0, this->dimSub[0]);
        (lhs->northEast)->Negate(0, (lhs->northEast)->Rank());
        
        // compute final component
        temp = ellCheckOne->RMultPart(lhs->southWest, 0, 0, this->dimSub[1]);
        delete ellCheckOne;
        (lhs->northWest)->SubAssignLowRank(temp);
        delete temp;
        
        // return answer
        return lhs;
    }
}

// make a dense version
template <class T> void HMat<T>::MakeDense(T** src, int &ldimIn) const
{
    // input check
    if (src == NULL)
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:MakeDense:badInput",
                           "Input array is NULL.");
        
    // allocate for answer
    T *ans = (T*)mxMalloc(sizeof(T) * this->m * this->n);
    ldimIn = this->m;
    
    // if matrix is dense
    if (this->IsDense() == true)
    {
        for (int j = 0; j < this->n; j++)
            memcpy(ans + j*ldimIn, this->data + j*this->ldim, sizeof(T) * this->m);
    }
    else
    {
        // northwest corner
        int lds;
        T* temp = NULL;
        (this->northWest)->MakeDense(&temp, lds);
        for (int j = 0; j < dimSub[1]; j++)
            memcpy(ans + j*ldimIn, temp + j*lds, sizeof(T) * dimSub[0]);
        mxFree(temp);
        temp = NULL;
        
        // southwest corner
        (this->southWest)->MakeDense(&temp, lds);
        for (int j = 0; j < dimSub[1]; j++)
            memcpy(ans + dimSub[0] + j*ldimIn, temp + j*lds, sizeof(T) * dimSub[2]);
        mxFree(temp);
        temp = NULL;
        
        // northeast corner
        (this->northEast)->MakeDense(&temp, lds);
        for (int j = 0; j < dimSub[3]; j++)
            memcpy(ans + (j + dimSub[1])*ldimIn, temp + j*lds, sizeof(T) * dimSub[0]);
        mxFree(temp);
        temp = NULL;
        
        // southeast corner
        (this->southEast)->MakeDense(&temp, lds);
        for (int j = 0; j < dimSub[3]; j++)
            memcpy(ans + (j + dimSub[1])*ldimIn + dimSub[0], temp + j* lds, sizeof(T) * dimSub[2]);
        mxFree(temp);
        temp = NULL;
    }
    (*src) = ans;
}

// returns storage in hierarhical form
template <class T> void HMat<T>::GenerateOutput(T** outData, int** outMeta, int& totalDataSize, int& nrows) const
{
    // input validation
    if ((outData == NULL) || (outMeta == NULL))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:GenerateOutput:badInput",
                          "One or more essential input arrays are NULL.");
    
    // get total storage amount
    nrows = 0;
    totalDataSize = this->Storage(nrows);
    
    // allocate for return data
    (*outData) = (T*)mxMalloc(sizeof(T) * totalDataSize);
    memset(*outData, 0, sizeof(T) * totalDataSize);
    (*outMeta) = (int*)mxMalloc(sizeof(int) * nrows * NUMMETA);
    memset(*outMeta, 0, sizeof(int) * nrows * NUMMETA);
    
    // collect data
    int dataOffset = 0, metaIndex = 0;
    this->PlaceOutput(*outData, *outMeta, dataOffset, metaIndex);
}

// places storage into a hierarchical array
template <class T> void HMat<T>::PlaceOutput(T* outData, int* outMeta, int &dataOffset, int &metaIndex) const
{
    // input validation
    if ((outData == NULL) || (outMeta == NULL))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:PlaceOutput:badInput",
                          "One or more essential input arrays are NULL.");
    if ((dataOffset < 0) || (metaIndex < 0))
        mexErrMsgIdAndTxt("MATLAB:hmat:HMat:PlaceOutput:badIndex",
                          "One or more specified indices are out of range.");
    
    // check if this matrix is dense
    if (this->IsDense() == true)
    {
        // copy dense data
        for (int i = 0; i < this->n; i++)
            memcpy(outData + dataOffset + i*this->m, this->data + i*this->ldim, sizeof(T) * this->m);

        // update meta data
        // indicate a dense matrix
        outMeta[metaIndex*NUMMETA] = 0;
        outMeta[metaIndex*NUMMETA+1] = this->m;
        outMeta[metaIndex*NUMMETA+2] = this->n;
        outMeta[metaIndex*NUMMETA+3] = 0;
        outMeta[metaIndex*NUMMETA+4] = 0;
        outMeta[metaIndex*NUMMETA+5] = dataOffset;
        outMeta[metaIndex*NUMMETA+6] = -1;
        outMeta[metaIndex*NUMMETA+7] = -1;
        outMeta[metaIndex*NUMMETA+8] = -1;
        outMeta[metaIndex*NUMMETA+9] = imin(this->m, this->n);
        outMeta[metaIndex*NUMMETA+10] = -1;
        
        // update the offsets
        dataOffset += (this->m * this->n);
        metaIndex++;
    }
    else
    {
        int midx = metaIndex;
        
        // set some values of the meta data array
        outMeta[midx*NUMMETA] = 1;
        memcpy(outMeta + midx*NUMMETA + 1, this->dimSub, sizeof(int) * 4);
        
        // copy low-rank data from northeast corner
        int neRank = (this->northEast)->Rank();
        outMeta[midx*NUMMETA + 9] = neRank;
        (this->northEast)->CopyData(outData + dataOffset, this->dimSub[0],
                                    outData + dataOffset + this->dimSub[0]*neRank, this->dimSub[3]);
        
        // update the offset
        outMeta[midx*NUMMETA + 5] = dataOffset; 
        dataOffset += ((this->northEast)->Storage());
        
        // copy low-rank data from southwest corner
        int swRank = (this->southWest)->Rank();
        outMeta[midx*NUMMETA + 10] = swRank;
        (this->southWest)->CopyData(outData + dataOffset, this->dimSub[2],
                                    outData + dataOffset + this->dimSub[2]*swRank, this->dimSub[1]);
        
        // update the offset
        outMeta[midx*NUMMETA + 6] = dataOffset;
        dataOffset += ((this->southWest)->Storage());
        
        // northwest corner
        metaIndex++;
        outMeta[midx*NUMMETA + 7] = metaIndex;
        (this->northWest)->PlaceOutput(outData, outMeta, dataOffset, metaIndex);
        
        // southwest corner
        outMeta[midx*NUMMETA + 8] = metaIndex;
        (this->southEast)->PlaceOutput(outData, outMeta, dataOffset, metaIndex);
    }
}

// validate metadata
template <class T> bool HMat<T>::ValidateMetadata(int *metaData, const int nRows, const int maxLen) const
{
    // basic checks
    if ((metaData == NULL) || (nRows < 1) || (maxLen < 1))
        return false;
    
    // more advanced
    bool isValid = true;
    for (int i = 0; i < nRows; i++)
    {
        // data is low-rank
        if (metaData[i*NUMMETA] == 1)
        {
            // check subdivision sizes
            for (int j = 1; j <= 4; j++)
            {
                if (metaData[i*NUMMETA+j] < 0)
                {
                    mexPrintf("ValidateMetadata -- Invalid subdivison size (%d).\n", i);
                    isValid = false;
                }
            }
            // check storage locations
            for (int j = 5; j <= 6; j++)
            {
                if ((metaData[i*NUMMETA+j] < 0) || (metaData[i*NUMMETA+j] >= maxLen))
                {
                    mexPrintf("ValidateMetadata -- Storage location is larger than data array (%d).\n", i);
                    isValid = false;
                }
            }
            // check child indices
            for (int j = 7; j <= 8; j++)
            {
                if ((metaData[i*NUMMETA+j] < 0) || (metaData[i*NUMMETA+j] >= nRows))
                {
                    mexPrintf("ValidateMetadata -- Child indices are invalid (%d).\n", i);
                    isValid = false;
                }
            }
            // check rank
            int rankMaxNE = imin(metaData[i*NUMMETA+1], metaData[i*NUMMETA+4]);
            int rankMaxSW = imin(metaData[i*NUMMETA+2], metaData[i*NUMMETA+3]);
            if (metaData[i*NUMMETA+9] < 0)
            {
                mexPrintf("ValidateMetadata -- Rank is invalid [rank %d for size %d x %d] (%d).\n", metaData[i*NUMMETA+9],
                          metaData[i*NUMMETA+1], metaData[i*NUMMETA+4], i);
                isValid = false;
            }
            if (metaData[i*NUMMETA+10] < 0)
            {
                mexPrintf("ValidateMetadata -- Rank is invalid [rank %d size %d x %d] (%d).\n", metaData[i*NUMMETA+10],
                          metaData[i*NUMMETA+3], metaData[i*NUMMETA+2], i);
                isValid = false;
            }
        }
        else
        {
            // data is dense
            if (metaData[i*NUMMETA] == 0)
            {
                // check matrix size
                for (int j = 1; j <= 2; j++)
                {
                    if (metaData[i*NUMMETA+j] < 0)
                    {
                        mexPrintf("ValidateMetadata -- Matrix size is invalid (%d)\n", i);
                        isValid = false;
                    }
                }
                // check storage location
                if ((metaData[i*NUMMETA+5] < 0) || (metaData[i*NUMMETA+5] >= maxLen))
                {
                    mexPrintf("ValidateMetadata -- Storage location is larger than data array (%d).\n", i);
                    isValid = false;
                }
                
            }
            // incorrect specifier
            else
            {
                mexPrintf("ValidateMetadata -- Invalid specifier (%d).\n", i);
                isValid = false;
            }
        }
        // break if we can already tell the meta data is bad
        if (isValid == false)
            break;
    }
    return isValid;   
}
