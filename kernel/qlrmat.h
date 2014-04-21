/*
    @(#)File:                /kernel/qlrmat.h
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        21 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Declares routines for quad low-rank matrices
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
template <class T> class QuadLowRankMat {

    public:
        
        // default initialization
        QuadLowRankMat()
        {
            this->northWest = NULL;
            this->northEast = NULL;
            this->southWest = NULL;
            this->southEast = NULL;
            this->base = NULL;
            this->m = 0;
            this->n = 0;
            this->minDim = 0;
            memset(this->dimSub, 0, sizeof(int) * 4);
        }
        
        // destructor
        ~QuadLowRankMat()
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
            this->m = 0;
            this->n = 0;
            this->minDim = 0;
            this->base = NULL;
            memset(this->dimSub, 0, sizeof(int) * 4);
        }
        
        // copy constructor
        QuadLowRankMat(QuadLowRankMat const& rhs)
        {
            this->northWest = NULL;
            this->northEast = NULL;
            this->southWest = NULL;
            this->southEast = NULL;
            this->base = NULL;
            memset(this->dimSub, 0, sizeof(int) * 4);
                
            this->m = base.m;
            this->n = base.n;
            this->minDim = rhs.minDim;
                
            if (rhs.base != NULL)
            {
                this->base = new LowRankMat<T>(rhs.base);
                this->dimSub[0] = rhs.dimSub[0];
                this->dimSub[1] = rhs.dimSub[1];
            }
            else
            {
                this->northWest = new QuadLowRankMat<T>(*(rhs.northWest));
                this->southEast = new QuadLowRankMat<T>(*(rhs.southEast));
                this->northEast = new QuadLowRankMat<T>(*(rhs.northEast));
                this->southWest = new QuadLowRankMat<T>(*(rhs.southWest));
                memcpy(this->dimSub, rhs.dimSub, sizeof(int) * 4);
            }
        }
        
        // constructor from individual generators
        QuadLowRankMat(T* left, const int ldl, const int inM,
                       T* right, const int ldr, const int inN, const int rIn, const int lim)
        {
            // input validation
            if ((left == NULL) || (right == NULL))
            {
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:badInput",
                                  "One or more supplied input data arrays are NULL.");
            }
            if ((ldl < inM) || (ldr < inN))
            {
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:badInput",
                    "One or more supplied leading dimensions are less than matrix dimensions.");
            }
            if (rIn < 1)
            {
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:badInput",
                    "Input rank is an invalid value.");
            }
            
            this->northWest = NULL;
            this->northEast = NULL;
            this->southWest = NULL;
            this->southEast = NULL;
            this->base = NULL;
            memset(this->dimSub, 0, sizeof(int) * 4);
                
            this->m = mIn;
            this->n = nIn;
            this->minDim = lim;
            
            // base case
            if ((inM <= lim) || (inN <= lim))
            {
                this->base = new LowRankMat<T>(left, ldl, inM, right, ldr, inN);
                this->dimSub[0] = mIn;
                this->dimSub[1] = nIn;
                // force base to compress
                (this->base)->CompressVectors();
            }
            else
            {
                int mh = (int)ceil(mIn / 2.0), nh = (int)ceil(nIn / 2.0);
                this->dimSub[0] = imin(mh, nh);
                this->dimSub[1] = this->dimSub[0];
                this->dimSub[2] = mIn - this->dimSub[0];
                this->dimSub[3] = nIn - this->dimSub[1];
                
                this->northWest = new QuadLowRankMat<T>(left, ldl, this->dimSub[0],
                                                         right, ldr, this->dimSub[1], rIn, lim);
                this->northEast = new QuadLowRankMat<T>(left, ldl, this->dimSub[0],
                                                         right + this->dimSub[1], ldr, this->dimSub[3], rIn, lim);
                this->southWest = new QuadLowRankMat<T>(left + this->dimSub[0], ldl, this->dimSub[2],
                                                         right, ldr, this->dimSub[1], rIn, lim);
                this->southEast = new QuadLowRankMat<T>(left + this->dimSub[0], ldl, this->dimSub[2],
                                                         right + this->dimSub[1], ldr, this->dimSub[3], rIn, lim);
            }
        }
        
        // raw data constructor
        QuadLowRankMat(T* source, const int ldIn, const int mIn, const int nIn, const int lim)
        {
            this->northWest = NULL;
            this->northEast = NULL;
            this->southWest = NULL;
            this->southEast = NULL;
            this->base = NULL;
            this->minDim = lim;
            memset(this->dimSub, 0, sizeof(int) * 4);

            // input validation
            if (source == NULL)
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:badInput",
                    "Input data array is NULL.");
            if ((mIn < 1) || (nIn < 1))
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:badInput",
                    "Specified dimensions of input data array are invalid.");
            if (ldIn < mIn)
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:badInput",
                    "Specified leading dimension of data out of range.");
            if (lim < 1)
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:badInput",
                    "Bad limit specifier supplied.");
                
            this->m = mIn;
            this->n = nIn;
                
            // store matrix as dense
            if (imax(mIn, nIn) <= lim)
            {
                this->base = new LowRankMat<T>(source, ldIn, mIn, nIn);
                this->dimSub[0] = mIn;
                this->dimSub[1] = nIn;
            }
            // store matrix as hierarchical
            else
            {
                int mh = (int)ceil(mIn / 2.0), nh = (int)ceil(nIn / 2.0);
                this->dimSub[0] = imin(mh, nh);
                this->dimSub[1] = this->dimSub[0];
                this->dimSub[2] = mIn - this->dimSub[0];
                this->dimSub[3] = nIn - this->dimSub[1];
                
                // create quad low-rank matrices
                this->northEast = new QuadLowRankMat<T>(source + this->dimSub[1]*ldIn, ldIn,
                                            this->dimSub[0], this->dimSub[3], lim);
                this->southWest = new QuadLowRankMat<T>(source + this->dimSub[0], ldIn,
                                            this->dimSub[2], this->dimSub[1], lim);
                this->northWest = new QuadLowRankMat<T>(source, ldIn,
                                            this->dimSub[0], this->dimSub[1], lim);
                this->southWest = new QuadLowRankMat<T>(source + this->dimSub[0] + this->dimSub[1]*ldIn, ldIn,
                                            this->dimSub[2], this->dimSub[3], lim);
                
                // check whether it's worth it or if we should just make this matrix low-rank
                // need to find a good way to do this...
            }
        }
        
        // returns matrix dimensions
        void Dims(int *a) const
        {
            if (a == NULL)
            {
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:Dims:badInput",
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
                mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:DimsSub:badInput",
                                  "Supplied vector to put dimensions into is NULL.");
            }
            else
                memcpy(a, this->dimSub, sizeof(int) * 4);
            return;
        }
        
        // returns whether matrix is actually a single low-rank matrix
        bool IsBase() const
        {
            return (this->base != NULL);
        }
        
        // assignment operator
        QuadLowRankMat<T>& operator=(const QuadLowRankMat<T> rhs)
        {
            rhs.Swap(*this);
            return *this;
        }
        
        /************* operator definitions *************/
        
        // add and assign
        QuadLowRankMat<T>& operator += (const QuadLowRankMat<T>);
        
        // subtract and assign
        QuadLowRankMat<T>& operator -= (const QuadLowRankMat<T>);
        
        // add two hierarchical matrices
        QuadLowRankMat<T> operator + (const QuadLowRankMat<T>) const ;
        
        // subtract two hierarchical matrices
        QuadLowRankMat<T> operator - (const QuadLowRankMat<T>) const ;
        
        /************* specialized addition *************/
        
        // subtract and assign a submatrix of another low-rank matrix
        void SubAssignPart(QuadLowRankMat<T>*, int, int);
        
        /************* specialized multiplication *************/
        
        // right-multiply by a submatrix of another low-rank matrix
        QuadLowRankMat<T>* RMult(QuadLowRankMat<T>*) const;
        
        // left-multiply by a submatrix of another low-rank matrix
        QuadLowRankMat<T>* LMult(QuadLowRankMat<T>*) const;
        
        // right-multiply and assign by a dense matrix
        void RMultAssign(T *rhs, const int nRows, const int ldim)
        
        // left-multiply and assign by a dense matrix
        void LMultAssign(T *rhs, const int nRows, const int ldim)
        
        /************* other *************/
        
        // negates the entries
        void Negate()
        {
            if (this->IsBase())
                (this->base)->Negate();
            else
            {
                (this->northWest)->Negate();
                (this->northEast)->Negate();
                (this->southWest)->Negate();
                (this->southEast)->Negate();
            }
        }
        
        // returns the total amount of storage required for this matrix
        int Storage(int &nsub) const
        {
            int storage = 0;
            if (this->IsBase())
            {
                storage = (this->base)->Storage();
                nsub++;
            }
            else
            {
                nsub += 3;
                storage = (this->northWest)->Storage(nsub) + (this->southEast)->Storage(nsub) +
                          (this->southWest)->Storage(nsub) + (this->northEast)->Storage(nsub);
            }
            return storage;
        }
    
        // return maximum rank
        int MaxRank() const
        {
            int rankVal = 0;
            if (this->IsBase())
            {
                if ((this->base) != NULL)
                    rankVal = (this->base)->Rank();
            }
            else
            {
                int rankVal = 0;
                if ((this->northWest) != NULL)
                    rankVal += (this->northWest)->MaxRank();
                if ((this->northEast) != NULL)
                    rankVal += (this->northEast)->MaxRank();
                if ((this->southWest) != NULL)
                    rankVal += (this->southWest)->MaxRank();
                if ((this->southEast) != NULL)
                    rankVal += (this->southEast)->MaxRank();
            }
            return rankVal;
        }
    
        // copies data from this structure into a vector
        void CopyData(T*, int, T*, int) const;
        
        // copies data from this structure into a vector
        void CopyLeftData(T*, int) const;
        
        // copies data from this structure into a vector
        void CopyRightData(T*, int) const;
    
        // convert to a low-rank matrix
        LowRankMat<T>* Convert() const;
    
        // make a dense version of the matrix
        void MakeDense(T**, int&);
    
        // make a dense version of the matrix
        void MakeDense(T**, int&, int, int, int, int);
    
    private:
        
        // quad-tree children
        QuadLowRankMat<T> *northWest, *southEast, *northEast, *southWest;
        
        // should only be non-NULL if this is the finest level of the tree
        LowRankMat<T> *base;
        
        // matrix dimensions
        int dimSub[4];
        int m, n;
        
        // minimum dim
        int minDim;
        
        // swap data elements
        void Swap(HMat &s)
        {
            std::swap(this->northWest, s.northWest);
            std::swap(this->southEast, s.southEast);
            std::swap(this->northEast, s.northEast);
            std::swap(this->southWest, s.southWest);
            std::swap(this->base, s.base);
            std::swap(this->dimSub, s.dimSub);
            std::swap(this->m, s.m);
            std::swap(this->n, s.n);
            std::swap(this->minDim, s.minDim);
        } 
};

// compare the dimensions of two hierarchical matrices
template <class T> bool DimMatch(QuadLowRankMat<T>* const rhs1, QuadLowRankMat<T>* const rhs2)
{
    // input validation
    if ((rhs1 == NULL) || (rhs2 == NULL))
        mexErrMsgIdAndTxt("MATLAB:qlrmat:DimMatch:invalidInput",
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
template <class T> QuadLowRankMat<T>& QuadLowRankMat<T>::operator += (const QuadLowRankMat<T> rhs)
{
    // check dimension match
    if ((rhs.m!= this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:PlusEqual:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // next check if either side (or both) is actually dense
    if (this->IsBase() || rhs.IsBase())
    {
        // both matrices are dense - simple add
        if (this->IsDense() && rhs.IsDense())
            this->base += (*(rhs.base));
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:PlusEqual:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the quad structures are different
        (*(this->northWest)) += (*(rhs.northWest));
        (*(this->southEast)) += (*(rhs.southEast));
        (*(this->northEast)) += (*(rhs.northEast));
        (*(this->southWest)) += (*(rhs.southWest));
    }
}

// subtraction-assignment of hierarhical matrices
template <class T> QuadLowRankMat<T>& QuadLowRankMat<T>::operator -= (const QuadLowRankMat<T> rhs)
{
    // check dimension match
    if ((rhs.m!= this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:MinusEqual:dimMismatch",
                          "Matrix dimensions are not equal.");
        
    // next check if either side (or both) is actually dense
    if (this->IsBase() || rhs.IsBase())
    {
        // both matrices are dense - simple add
        if (this->IsDense() && rhs.IsDense())
            this->base -= (*(rhs.base));
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:MinusEqual:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the quad structures are different
        (*(this->northWest)) -= (*(rhs.northWest));
        (*(this->southEast)) -= (*(rhs.southEast));
        (*(this->northEast)) -= (*(rhs.northEast));
        (*(this->southWest)) -= (*(rhs.southWest));
    }
}

// add two hierarchical matrices to form a new instance
template <class T> QuadLowRankMat<T> QuadLowRankMat<T>::operator + (const QuadLowRankMat<T> rhs) const
{
    // check dimension match
    if ((rhs.m!= this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:Plus:dimMismatch",
                          "Matrix dimensions are not equal.");
    
    // initialize solution
    QuadLowRankMat<T> lhs(rhs);
    lhs += rhs;
    
    // return
    return lhs;
}

// subtract two hierarchical matrices to form a new instance
template <class T> QuadLowRankMat<T> QuadLowRankMat<T>::operator - (const QuadLowRankMat<T> rhs) const
{
    // check dimension match
    if ((rhs.m!= this->m) || (rhs.n != this->n))
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:Plus:dimMismatch",
                          "Matrix dimensions are not equal.");
    
    // initialize solution
    QuadLowRankMat<T> lhs(rhs);
    lhs -= rhs;
    
    // return
    return lhs;
}

// multiplication-assignment of low-rank matrices
template <class T> QuadLowRankMat<T>& QuadLowRankMat<T>::operator *= (const QuadLowRankMat<T> rhs)
{
    // check dimension match
    if (this->n != rhs.m)
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:TimesEqual:dimMismatch",
                          "Inner matrix dimensions are not equal.");
        
    // next check if either side (or both) is actually dense
    if (this->IsBase() || rhs.IsBase())
    {
        // both matrices are dense - simple add
        if (this->IsBase() && rhs.IsBase())
            this->base *= (*(rhs.base));
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:PlusEqual:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the quad structures are different
        QuadLowRankMat<T> *temp = new QuadLowRankMat<T>(*(this->northEast));
        
        // compute northeast component
        (*(this->northEast)) *= (*(rhs.southEast));
        (*(this->northEast)) += ((*(this->northWest)) * (*(rhs.northEast)));
        
        // compute northwest component
        (*(this->northWest)) *= (*(rhs.northWest));
        (*(this->northWest)) += ((*temp) * (*(rhs.southWest)));
        
        delete temp;
        temp = new QuadLowRankMat<T>(*(this->southWest));
        
        // compute southwest component
        (*(this->southWest)) *= (*(rhs.northWest));
        (*(this->southWest)) += ((*(this->southEast)) * (*(rhs.southWest)));
        
        // compute southeast component
        (*(this->southEast)) *= (*(rhs.southEast));
        (*(this->southEast)) += ((*temp) * (*(rhs.northEast)));
    }
    return (*this);
}

// multiply two low-rank matrices to form a new instance
template <class T> QuadLowRankMat<T> QuadLowRankMat<T>::operator * (const QuadLowRankMat<T> rhs) const
{
    // check dimension match
    if (this->n != rhs.m)
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:TimesEqual:dimMismatch",
                          "Inner matrix dimensions are not equal.");
        
    // initialize empty result
    LowRankMat<T> lhs();
    lhs.m = this->m;
    lhs.n = rhs.n;
    
    // next check if either side (or both) is actually dense
    if (this->IsBase() || rhs.IsBase())
    {
        // both matrices are dense - simple add
        if (this->IsBase() && rhs.IsBase())
        {
            lhs.base = (*(this->base)) * (*(rhs.base));
            lhs.dimSub[0] = this->m;
            lhs.dimSub[1] = rhs.n;
            lhs.m = this->m;
            lhs.n = rhs.n;
        }
        // this case must be invalid for now
        else
        {
            mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:PlusEqual:badLayout",
                "Input hierarchical matrices have different decompositions.");
        }
    }
    else
    {
        // this will toss an error if the quad structures are different
        // compute northwest component
        lhs.northWest = ((*(this->northWest)) * (*(rhs.northWest))) +
                        ((*(this->northEast)) * (*(rhs.southWest)));
        lhs.northEast = ((*(this->northWest)) * (*(rhs.northEast))) +
                        ((*(this->northEast)) * (*(rhs.southEast)));
        lhs.southWest = ((*(this->southWest)) * (*(rhs.northWest))) +
                        ((*(this->southEast)) * (*(rhs.southWest)));
        lhs.southEast = ((*(this->southWest)) * (*(rhs.northEast))) +
                        ((*(this->southEast)) * (*(rhs.southEast)));
    }
    return lhs;
}

/************* specialized addition *************/`

// subtract and assign a submatrix of another low-rank matrix
template <class T> void QuadLowRankMat<T>::SubAssignPart(QuadLowRankMat<T> *rhs, const int rI, const int cI)
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

/************* specialized multiplication *************/

// right-multiply by a submatrix of another low-rank matrix
template <class T> QuadLowRankMat<T>* QuadLowRankMat<T>::RMult(QuadLowRankMat<T> *rhs) const
{
    // check null condition
    if (rhs == NULL)
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:RMult:badInput",
                          "Input matrix is NULL.");
    
    // compare sizes
    if (this->n != rhs->m)
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:RMult:dimMismatch",
                          "Inner matrix dimensions are not equal.");
        
    // check if subdimensions are ok
    if ((this->dimSub[1] != rhs->dimSub[0]) || (this->dimSub[3] != rhs->dimSub[2]))
    {
        // convert to low-rank matrix
        LowRankMat<T> *op1 = this->Convert();
        LowRankMat<T> *op2 = rhs->Convert();
        op1 *= op2;
        delete op2;
        
        QuadLowRankMat<T> *lhs = new QuadLowRankMat<T>();
        lhs->minDim = imin(rhs->minDim, this->minDim);
        lhs->base = op1;
        lhs->m = this->m;
        lhs->n = rhs->n;
        lhs->dimSub[0] = this->m;
        lhs->dimSub[1] = this->n;
        return lhs;
    }
    else
    {
        // otherwise, return multiplication of the quad mats
        QuadLowRankMat<T> *lhs = new QuadLowRankMat<T>();
        (*lhs) = (*this) * (*rhs);
        return lhs;
    }
}
        
        // left-multiply by a submatrix of another low-rank matrix
template <class T> QuadLowRankMat<T>* QuadLowRankMat<T>::LMult(QuadLowRankMat<T> *rhs) const
{
    // check null condition
    if (rhs == NULL)
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:RMult:badInput",
                          "Input matrix is NULL.");
    
    // compare sizes
    if (this->n != rhs->m)
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:RMult:dimMismatch",
                          "Inner matrix dimensions are not equal.");
        
    // check if subdimensions are ok
    if ((this->dimSub[1] != rhs->dimSub[0]) || (this->dimSub[3] != rhs->dimSub[2]))
    {
        // convert to low-rank matrix
        LowRankMat<T> *op1 = this->Convert();
        LowRankMat<T> *op2 = rhs->Convert();
        op2 *= op1;
        delete op1;
        
        QuadLowRankMat<T> *lhs = new QuadLowRankMat<T>();
        lhs->base = op2;
        lhs->minDim = imin(this->minDim, rhs->minDim);
        lhs->m = this->m;
        lhs->n = rhs->n;
        lhs->dimSub[0] = this->m;
        lhs->dimSub[1] = this->n;
        return lhs;
    }
    else
    {
        // otherwise, return multiplication of the quad mats
        QuadLowRankMat<T> *lhs = new QuadLowRankMat<T>();
        (*lhs) = (*rhs) * (*this);
        return lhs;
    }
}

// left-multiply and assign by a dense matrix
template <class T> void QuadLowRankMat<T>::RMultAssign(T *rhs, const int nCols, const int ldim)
{
    // check input column
    if (nCols < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:RMultAssign:badNumCols",
                          "Specified number of columns is invalid.");
    }  
    if (ldim < this->m)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:RMultAssign:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    // base case
    if (this->IsBase() == true)
    {
        (this->base)->RMultAssignPart(rhs, nCols, ldim);
        this->dimSub[1] = nCols;
        this->n = nCols;
    }
    // quad case
    else
    {
        // check to see if resulting matrix is too small
        if (nCols <= imin(this->minDim, imax(this->dimSub[1], this->dimSub[3])))
        {
            this->base = this->Convert();
            (this->base)->RMultAssignPart(rhs, nRows, ldim);
            this->n = nCols;
            this->dimSub[0] = this->m;
            this->dimSub[1] = nCols;
            this->dimSub[2] = 0;
            this->dimSub[3] = 0;
            delete (this->northWest);
            this->northWest = NULL;
            delete (this->northEast);
            this->northEast = NULL;
            delete (this->southWest);
            this->southWest = NULL;
            delete (this->southEast);
            this->southEast = NULL;
        }
        else
        {
            // compute partitioning
            int nc1 = this->dimSub[1];
            int nc2 = nCols - nc1;
            
            // compute northern portion
            QuadLowRankMat<T> *temp1 = new QuadLowRankMat<T>(*(this->northWest));
            QuadLowRankMat<T> *temp2 = new QuadLowRankMat<T>(*(this->northEast));
            // northwest
            (this->northWest)->RMultAssignPart(rhs, nc1, ldim);
            temp2->RMultAssignPart(rhs + this->dimSub[1], nc1, ldim);
            (*(this->northWest)) += (*temp2);
            delete temp2;
            // southwest
            (this->northEast)->RMultAssignPart(rhs + this->dimSub[1] + nc1*ldim, nc2, ldim);
            temp1->RMultAssignPart(rhs + nc1*ldim, nc2, ldim);
            (*(this->northEast)) += (*temp1);
            delete temp1;
            
            // compute eastern portion
            temp1 = new QuadLowRankMat<T>(*(this->southWest));
            temp2 = new QuadLowRankMat<T>(*(this->southEast));
            // northwest
            (this->southWest)->RMultAssignPart(rhs, nc1, ldim);
            temp2->RMultAssignPart(rhs + this->dimSub[1], nc1, ldim);
            (*(this->southWest)) += (*temp2);
            delete temp2;
            // southwest
            (this->southEast)->RMultAssignPart(rhs + this->dimSub[1] + nc1*ldim, nc2, ldim);
            temp1->RMultAssignPart(rhs + nc1*ldim, nc2, ldim);
            (*(this->southEast)) += (*temp1);
            delete temp1;
            
            this->dimSub[1] = nc1;
            this->dimSub[3] = nc2;
            this->n = nCols;
        }
    }
}

// left-multiply and assign by a dense matrix
template <class T> void QuadLowRankMat<T>::LMultAssign(T *rhs, const int nRows, const int ldim)
{
    // check input column
    if (nRows < 1)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:LMultAssign:badNumCols",
                          "Specified number of rows is invalid.");
    }  
    if (ldim < nRows)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:LMultAssign:dimMismatch",
                          "Leading dimension of input data is invalid.");
    }
    
    // base case
    if (this->IsBase() == true)
    {
        (this->base)->LMultAssignPart(rhs, nRows, ldim);
        this->dimSub[0] = nRows;
        this->m = nRows;
    }
    // quad case
    else
    {
        // check to see if resulting matrix is too small
        if (nRows <= imin(this->minDim, imax(this->dimSub[0], this->dimSub[2])))
        {
            this->base = this->Convert();
            (this->base)->LMultAssignPart(rhs, nRows, ldim);
            this->m = nRows;
            this->dimSub[0] = nRows;
            this->dimSub[1] = this->n;
            this->dimSub[2] = 0;
            this->dimSub[3] = 0;
            delete (this->northWest);
            this->northWest = NULL;
            delete (this->northEast);
            this->northEast = NULL;
            delete (this->southWest);
            this->southWest = NULL;
            delete (this->southEast);
            this->southEast = NULL;
        }
        else
        {
            // compute partitioning
            int nr1 = this->dimSub[0];
            int nr2 = nRows - nr1;
            
            // compute western portion
            QuadLowRankMat<T> *temp1 = new QuadLowRankMat<T>(*(this->northWest));
            QuadLowRankMat<T> *temp2 = new QuadLowRankMat<T>(*(this->southWest));
            // northwest
            (this->northWest)->LMultAssignPart(rhs, nr1, ldim);
            temp2->LMultAssignPart(rhs + this->dimSub[1]*ldim, nr1, ldim);
            (*(this->northWest)) += (*temp2);
            delete temp2;
            // southwest
            (this->southWest)->LMultAssignPart(rhs + this->dimSub[1]*ldim + nr1, nr2, ldim);
            temp1->LMultAssignPart(rhs + nr1, nr2, ldim);
            (*(this->southWest)) += (*temp1);
            delete temp1;
            
            // compute eastern portion
            temp1 = new QuadLowRankMat<T>(*(this->northEast));
            temp2 = new QuadLowRankMat<T>(*(this->southEast));
            // northwest
            (this->northEast)->LMultAssignPart(rhs, nr1, ldim);
            temp2->LMultAssignPart(rhs + this->dimSub[1]*ldim, nr1, ldim);
            (*(this->northEast)) += (*temp2);
            delete temp2;
            // southwest
            (this->southEast)->LMultAssignPart(rhs + this->dimSub[1]*ldim + nr1, nr2, ldim);
            temp1->LMultAssignPart(rhs + nr1, nr2, ldim);
            (*(this->southEast)) += (*temp1);
            delete temp1;
            
            this->dimSub[0] = nr1;
            this->dimSub[2] = nr2;
            this->m = nRows;
        }
    }
}

/************* other *************/

// copy data into an array
template <class T> void QuadLowRankMat<T>::CopyData(T* left, int ldl, T* right, int ldr) const
{
    // input validation
    if ((left == NULL) || (right == NULL))
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:CopyData:badInput",
                          "One or more supplied input data arrays are NULL.");
    }
    if ((ldl < this->m) || (ldr < this->n))
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:CopyData:badInput",
            "One or more supplied leading dimensions are less than matrix dimensions.");
    }
    
    // base
    if (this->IsBase() == true)
        (this->base)->CopyData(left, ldl, right, ldr);
    else
    {
        int rankV[3] = { (this->northWest)->MaxRank(), (this->northEast)->MaxRank(),
                         (this->southWest)->MaxRank() };
                         
        // copy data
        (this->northWest)->CopyData(left, ldl, right, ldr);
        (this->northEast)->CopyData(left + rankV[0]*ldl, ldl,
                                    right + rankV[0]*ldr + this->dimSub[1], ldr);
        (this->southWest)->CopyData(left + (rankV[0] + rankV[1])*ldl + this->dimSub[0], ldl,
                                    right + (rankV[0] + rankV[1])*ldr, ldr);
        (this->southEast)->CopyData(left + (rankV[0] + rankV[1] + rankV[2])*ldl + this->dimSub[0], ldl,
                                    right + (rankV[0] + rankV[1] + rankV[2])*ldr + this->dimSub[1], ldr);
    }
    return;
}

// copy left vectors into array
template <class T> void QuadLowRankMat<T>::CopyLeftData(T* left, int ldl) const
{
    // input validation
    if (left == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:CopyLeftData:badInput",
                          "Supplied input data array is NULL.");
    }
    if (ldl < this->m)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:CopyLeftData:badInput",
            "Supplied leading dimensions is less than matrix dimension.");
    }
    
    // base
    if (this->IsBase() == true)
    {
        int rankVal = (this->base)->Rank();
        (this->base)->CopyLeftData(left, ldl, 0, rankVal);
    }
    else
    {
        int rankV[3] = { (this->northWest)->MaxRank(), (this->northEast)->MaxRank(),
                         (this->southWest)->MaxRank() };
                         
        // copy data
        (this->northWest)->CopyLeftData(left, ldl);
        (this->northEast)->CopyLeftData(left + rankV[0]*ldl, ldl);
        (this->southWest)->CopyLeftData(left + (rankV[0] + rankV[1])*ldl + this->dimSub[0], ldl);
        (this->southEast)->CopyLeftData(left + (rankV[0] + rankV[1] + rankV[2])*ldl + this->dimSub[0]);
    }
    return;
}

// copy left vectors into array
template <class T> void QuadLowRankMat<T>::CopyRightData(T* right, int ldr) const
{
    // input validation
    if (right == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:CopyRightData:badInput",
                          "Supplied input data array is NULL.");
    }
    if (ldr < this->n)
    {
        mexErrMsgIdAndTxt("MATLAB:qlrmat:QuadLowRankMat:CopyRightData:badInput",
            "Supplied leading dimensions is less than matrix dimension.");
    }
    
    // base
    if (this->IsBase() == true)
    {
        int rankVal = (this->base)->Rank();
        (this->base)->CopyRightData(left, ldl, 0, rankVal);
    }
    else
    {
        int rankV[3] = { (this->northWest)->MaxRank(), (this->northEast)->MaxRank(),
                         (this->southWest)->MaxRank() };
                         
        // copy data
        (this->northWest)->CopyRightData(right, ldr);
        (this->northEast)->CopyRightData(right + rankV[0]*ldr + this->dimSub[1], ldr);
        (this->southWest)->CopyRightData(right + (rankV[0] + rankV[1])*ldr, ldr);
        (this->southEast)->CopyRightData(right + (rankV[0] + rankV[1] + rankV[2])*ldr + this->dimSub[1], ldr);
    }
    return;
}

// convert to a low-rank matrix
template <class T> LowRankMat<T>* QuadLowRankMat<T>::Convert() const
{
    if (this->IsBase() == true)
    {
        LowRankMat<T> *lhs = new LowRankMat<T>(*(this->base));
        return lhs;
    }
    else
    {
        int rankBound = this->MaxRank();
        
        // allocate for vectors
        T* left = mxMalloc(sizeof(T) * this->m * rankBound);
        memset(left, 0, sizeof(T) * this->m * rankBound);
        T* right = mxMalloc(sizeof(T) * this->n * rankBound);
        memset(right, 0, sizeof(T) * this->n * rankBound);
        
        // copy data
        this->CopyData(left, this->m, right, this->n);
        
        // create a low-rank structure
        LowRankMat<T> *lhs = new LowRankMat<T>(left, this->m, this->m, right, this->n, this->n, rankBound);
        mxFree(left);
        mxFree(right);
        return lhs;
    }
}

// make a dense version of the matrix
template <class T> void QuadLowRankMat<T>::MakeDense(T** src, int &ldim)
{
    this->MakeDense(src, ldim, 0, 0, this->m, this->n);
}

// make a dense version of the matrix
template <class T> void QuadLowRankMat<T>::MakeDense(T** src, int &ldim, const int rI, const int cI,
                                                 const int nRows, const int nCols)
{
    // check index ranges
    if ((rI < 0) || ((this->m - rI) < nRows) ||
        (cI < 0) || ((this->n - cI) < nCols))
        mexErrMsgIdAndTxt("MATLAB:lrmat:QuadLowRankMat:MakeDense:badIndices",
                          "Supplied submatrix indices are invalid.");
    
    // allocate dense array for data
    int mNew = nRows, nNew = nCols;
    T* dataCopy = NULL;
    
    // check if this is the base matrix
    if (this->IsBase())
        (this->base)->MakeDense(&dataCopy, ldim, rI, cI, nRows, nCols);
    else
    {
        // allocate for resulting data
        dataCopy = (T*)mxMalloc(sizeof(T) * nRows * nCols);
        ldim = nRows;
        
        // get sizes for each portion
        int sz[4];
        sz[0] = imin(dimSub[0] - rI, 0);
        sz[1] = imin(dimSub[1] - cI, 0);
        sz[2] = nRows - sz[0];
        sz[3] = nCols - sz[1];
        
        // northern portion
        if (sz[0] > 0)
        {
            // northwest
            if (sz[1] > 0)
            {
                T* temp = NULL;
                int ldt = 0;
                (this->northWest)->MakeDense(&temp, ldt, rI, cI, sz[0], sz[1]);
                for (int i = 0; i < sz[1]; i++)
                    memcpy(dataCopy + i*nRows, temp + i*ldt, sizeof(T) * sz[0]);
                mxFree(temp);
            }
            // northeast
            if (sz[3] > 0)
            {
                T* temp = NULL;
                int ldt = 0;
                (this->northEast)->MakeDense(&temp, ldt, rI, imax(cI, dimSub[1]), sz[0], sz[3]);
                for (int i = 0; i < sz[3]; i++)
                    memcpy(dataCopy + (sz[1]+i*nRows), temp + i*ldt, sizeof(T) * sz[0]);
                mxFree(temp);
            }
        }
        // southern portion
        if (sz[2] > 0)
        {
            // southwest
            if (sz[1] > 0)
            {
                T* temp = NULL;
                int ldt = 0;
                (this->southWest)->MakeDense(&temp, ldt, imax(rI, dimSub[0]), cI, sz[2], sz[1]);
                for (int i = 0; i < sz[1]; i++)
                    memcpy(dataCopy + sz[0] + i*nRows, temp + i*ldt, sizeof(T) * sz[2]);
                mxFree(temp);
            }
            // southeast
            if (sz[3] > 0)
            {
                T* temp = NULL;
                int ldt = 0;
                (this->southEast)->MakeDense(&temp, ldt, imax(rI, dimSub[0]), imax(cI, dimSub[1]), sz[2], sz[3]);
                for (int i = 0; i < sz[3]; i++)
                    memcpy(dataCopy + sz[0] + (sz[1] + i)*nRows, temp + i*ldt, sizeof(T) * sz[2]);
                mxFree(temp);
            }
        }
    }
    (*src) = dataCopy;
}
