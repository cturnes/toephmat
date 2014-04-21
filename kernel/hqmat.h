/*
    @(#)File:                /kernel/hqmat.h
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        21 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Declares routines for quad hierarchical matrices
*/
#pragma once

#include <cmath>
#include <cstdio>
#include <complex>
#include <climits>
#include "mex.h"
#include "qlrmat.h"
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
template <class T> class HQMat {

    public:
        
        // default initialization
        HQMat()
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
        ~HQMat()
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
        HQMat(HQMat const& rhs)
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
                    mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:badInput",
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
                    this->northWest = new HQMat<T>(*(rhs.northWest));
                if (rhs.southEast != NULL)
                    this->southEast = new HQMat<T>(*(rhs.southEast));
                if (rhs.northEast != NULL)
                    this->northEast = new QuadLowRankMat<T>(*(rhs.northEast));
                if (rhs.southWest != NULL)
                    this->southWest = new QuadLowRankMat<T>(*(rhs.southWest));
                memcpy(this->dimSub, rhs.dimSub, sizeof(int) * 4);
            }
        }
        
        // raw data constructor
        HQMat(T* source, const int ldIn, const int mIn, const int nIn, const int lim)
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
                mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:badInput",
                    "Input data array is NULL.");
            if ((mIn < 1) || (nIn < 1))
                mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:badInput",
                    "Specified dimensions of input data array are invalid.");
            if (ldIn < mIn)
                mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:badInput",
                    "Specified leading dimension of data out of range.");
            if (lim < 1)
                mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:badInput",
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
                this->northEast = new QuadLowRankMat<T>(source + this->dimSub[1]*ldIn, ldIn,
                                                    this->dimSub[0], this->dimSub[3], lim);
                this->southWest = new QuadLowRankMat<T>(source + this->dimSub[0], ldIn,
                                                    this->dimSub[2], this->dimSub[1], lim);
                // create hierarchical matrices
                this->northWest = new HQMat<T>(source, ldIn, this->dimSub[0], this->dimSub[1], lim);
                this->southEast = new HQMat<T>(source + this->dimSub[0] + this->dimSub[1]*ldIn, ldIn,
                                          this->dimSub[2], this->dimSub[3], lim);
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
        HQMat<T>& operator=(const HMat<T> rhs)
        {
            rhs.Swap(*this);
            return *this;
        }
        
        /************* operator definitions *************/
        
        // add and assign
        HQMat<T>& operator += (const HQMat<T>);
        
        // subtract and assign
        HQMat<T>& operator -= (const HQMat<T>);
        
        // add two hierarchical matrices
        HQMat<T> operator + (const HQMat<T>) const ;
        
        // subtract two hierarchical matrices
        HQMat<T> operator - (const HQMat<T>) const ;
        
        /************* specialized addition *************/

        // subtract a low-rank matrix from the hierarchical matrix
        void SubAssignLowRank(QuadLowRankMat<T>*,int,int);
        
        /************* specialized multiplication *************/
        
        // right-multiply by a low-rank matrix
        QuadLowRankMat<T>* RMultLowRank(QuadLowRankMat<T>*) const ;
        
        // left-multiply by a low-rank matrix
        QuadLowRankMat<T>* LMultLowRank(QuadLowRankMat<T>*) const ;
        
        /************* other *************/
        
        // apply the inverse to the left of an input
        void ApplyInverseLeft(T*, const int, const int) const;
        
        // compute the inverse
        HQMat<T>* Invert() const;
        
        // make a dense version
        void MakeDense(T**,int&) const;
        
    private:
        
        // quad-tree children
        HQMat<T> *northWest, *southEast;
        QuadLowRankMat<T> *northEast, *southWest;
        
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
template <class T> bool DimMatch(HQMat<T>* const rhs1, HQMat<T>* const rhs2)
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
template <class T> HQMat<T>& HQMat<T>::operator += (const HQMat<T> rhs)
{
    // check dimension match
    if ((rhs.m != this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:PlusEqual:dimMismatch",
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
            mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:PlusEqual:badLayout",
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
template <class T> HQMat<T>& HQMat<T>::operator -= (const HQMat<T> rhs)
{
    // check dimension match
    if ((rhs.m != this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:MinusEqual:dimMismatch",
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
            mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:MinusEqual:badLayout",
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
template <class T> HQMat<T> HQMat<T>::operator + (const HQMat<T> rhs) const
{
    // check dimension match
    if ((rhs.m != this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:Plus:dimMismatch",
                          "Matrix dimensions are not equal.");
    
    // initialize solution
    HQMat<T> lhs(rhs);
    
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
            mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:Plus:badLayout",
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
template <class T> HQMat<T> HQMat<T>::operator - (const HQMat<T> rhs) const
{
    // check dimension match
    if ((rhs.m != this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:Minus:dimMismatch",
                          "Matrix dimensions are not equal.");
    
    // initialize solution
    HQMat<T> lhs(rhs);
    
    // next check if either side (or both) is actually dense
    if (this->IsDense() || rhs.IsDense())
    {
        // both matrices are dense - simple add
        if (this->IsDense() && rhs.IsDense())
        {
            for (int j = 0; j < rhs.n; j++)
                for (int i = 0; i < rhs.m; i++)
                    lhs.data[i + rhs.m*j] -= this->data[i + this->ldim*j];
        }
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:Minus:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the hierarhical structures are different
        (*(lhs.northWest)) -= (*(this->northWest));
        (*(lhs.southEast)) -= (*(this->southEast));
        (*(lhs.northEast)) -= (*(this->northEast));
        (*(lhs.southWest)) -= (*(this->southWest));
    }
    return lhs;
}

/************* specialized addition *************/

// subtract a low-rank matrix to the hierarchical matrix
template <class T> void HQMat<T>::SubAssignLowRank(QuadLowRankMat<T>* rhs, const int rI=0, const int cI=0)
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
        rhs->MakeDense(&temp, rhsLdim, rI, cI, this->m, this->n);
        
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

/************* specialized multiplication *************/

// right-multiply by a low-rank matrix
template <class T> QuadLowRankMat<T>* HQMat<T>::RMultLowRank(QuadLowRankMat<T>* rhs) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check dimensions
    if (this->n != dimRhs[0])
    {
        mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:RMultLowRank:dimMismatch",
                          "Inner matrix dimensions must agree.");
    }
    
    // allocate for the matrix
    LowRankMat<T>* lhs = NULL;
    
    // check if the matrix is dense
    if (this->IsDense() == true)
    {
        // start by creating a matrix from the input matrix
        lhs = new LowRankMat<T>(*rhs);
        
        // then update it with a dense multiply
        lhs->LMultAssign(this->data, this->m, this->ldim); 
    }
    else
    {
        // get the left vectors from the rhs
        int rankV = rhs->MaxRank();
        T* temp = (T*)mxMalloc(sizeof(T) * this->n * rankV);
        rhs->CopyLeftData(temp, this->n);
        
        // get the new left vectors
        int ldout = 0;
        T* newLeft = this->RMultDense(temp, rankV, this->n, ldout);
        mxFree(temp);
        
        temp = (T*)mxMalloc(sizeof(T) * rhs->n * rankV);
        rhs->CopyRightData(temp, rhs->n);
        lhs = new QuadLowRankMat<T>(newLeft, ldout, this->m, temp, rhs->n, rhs->n, rankV);
        mxFree(temp);
    }
    return lhs;
}

// left-multiply by a low-rank matrix
template <class T> QuadLowRankMat<T>* HQMat<T>::LMultLowRank(QuadLowRankMat<T>* rhs) const
{
    // get matrix dimensions
    int dimRhs[2];
    rhs->Dims(dimRhs);
    
    // check dimensions
    if (this->m != dimRhs[1])
    {
        mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:LMultLowRank:dimMismatch",
                          "Inner matrix dimensions must agree.");
    }
    
    // allocate for the matrix
    LowRankMat<T>* lhs = NULL;
    
    // check if the matrix is dense
    if (this->IsDense() == true)
    {
        // start by creating a matrix from the input matrix
        lhs = new LowRankMat<T>(*rhs);
        
        // then update it with a dense multiply
        lhs->RMultAssign(this->data, this->n, this->ldim); 
    }
    else
    {
        // get the right vectors from the rhs
        int rankV = rhs->MaxRank();
        T* temp = (T*)mxMalloc(sizeof(T) * this->m * rankV);
        rhs->CopyRightData(temp, this->m);
        
        // get the new right vectors
        int ldout = 0;
        T* newRight = this->LMultDense(temp, rankV, this->m, ldout);
        mxFree(temp);
        
        temp = (T*)mxMalloc(sizeof(T) * rhs->m * rankV);
        rhs->CopyLeftData(temp, rhs->m);
        lhs = new QuadLowRankMat<T>(temp, rhs->m, rhs->m, newRight, ldout, rhs->n, rankV);
        mxFree(temp);
    }
    return lhs;
}

/************* other *************/

// compute the inverse
template <class T> HQMat<T>* HQMat<T>::Invert() const
{
    // input validation
    if (this->m != this->n)
    {
        mexWarnMsgIdAndTxt("MATLAB:hqmat:HQMat:Invert:singular",
                           "Matrix is singular and cannot be inverted.");
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
        HQMat<T> *lhs = new HQMat<T>(invData, order, order, order, order + 1);
        mxFree(invData);
        return lhs;
    }
    else
    {
        // initialize lhs structure
        HQMat<T> *lhs = new HQMat<T>();
        
        // set internal dimension variables
        lhs->m = this->m;
        lhs->n = this->n;
        memcpy(lhs->dimSub, this->dimSub, sizeof(int) * 4);
        
        // first invert northwest corner
        lhs->northWest = (this->northWest)->Invert();
        
        // compute special low rank matrices
        QuadLowRankMat<T> *ellCheckOne = (lhs->northWest)->RMultLowRank(this->northEast);
        QuadLowRankMat<T> *ellCheckTwo = (lhs->northWest)->LMultLowRank(this->southWest);
        
        // form the Schur complement
        HMat<T> *schur = new HMat<T>(*(this->southEast));
        LowRankMat<T> *temp = (this->southWest)->RMult(ellCheckOne);
        schur->SubAssignLowRank(temp);
        delete temp;
        
        // invert the Schur complement
        lhs->southEast = schur->Invert();
        delete schur;
        
        // compute the off-diagonal components
        lhs->southWest = (lhs->southEast)->RMultLowRank(ellCheckTwo);
        (lhs->southWest)->Negate();
        delete ellCheckTwo;
        lhs->northEast = (lhs->southEast)->LMultLowRank(ellCheckOne);
        (lhs->northEast)->Negate();
        
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
template <class T> void HQMat<T>::MakeDense(int &ldimIn) const
{
    // input check
    if (src == NULL)
        mexErrMsgIdAndTxt("MATLAB:hqmat:HQMat:MakeDense:badInput",
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

