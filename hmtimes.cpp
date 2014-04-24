/*
    @(#)File:                /hmtimes.cpp
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        20 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Hierarchical matrix multiplication
*/
#include <cstdlib>
#include <cstdio>
#include <complex>
#include <string.h>
#include "mex.h"
#include "lapack.h"
#include "blas.h"
#include "hmat.h"
#include "misc.h"

// main function
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // input validation
    if (nrhs != 2)
        mexErrMsgIdAndTxt("MATLAB:hmtimes:nrhs",
            "hmtimes ==> Incorrect number of inputs;\nSyntax is Y = hmtimes(H, X); (see help).");
    if (mxIsStruct(prhs[0]) == false)
        mexErrMsgIdAndTxt("MATLAB:hmtimes:noStructData",
            "hmtimes ==> Hierarchical matrix must be supplied as a struct (see help for dense2hm).");
    int dataFieldNumber = (int)mxGetFieldNumber(prhs[0], "data");
    int metaFieldNumber = (int)mxGetFieldNumber(prhs[0], "meta");
    if ((dataFieldNumber == -1) || (metaFieldNumber == -1))
        mexErrMsgIdAndTxt("MATLAB:hmtimes:noStructData",
            "hmtimes ==> Struct data for hierarchical matrix is missing one or both fields.");
    if (mxIsEmpty(prhs[1]) == true)
        mexErrMsgIdAndTxt("MATLAB:hmtimes:emptyInput",
            "hmtimes ==> Supplied vector X cannot be empty.");
        
    // check whether matrix is complex
    bool isComplex = IsHMatComplex(prhs[0]);
    
    // check whether overall computations are complex
    if (mxIsComplex(prhs[1]) == true)
        isComplex = true;
        
    int *meta = NULL, dataLen = 0, metaLen = 0;
    if (isComplex == true)
    {
        // read in struct data
        cmpx *data = NULL;
        readStructData(prhs[0], &data, &meta, dataLen, metaLen);
        
        // make a hierarchical matrix
        HMat<cmpx> *h = new HMat<cmpx>(data, meta, dataLen, metaLen);
        
        // check dimensions
        int hmatDims[2];
        h->Dims(hmatDims);
        if (hmatDims[1] != (int)mxGetM(prhs[1]))
        {
            delete h;
            mxFree(data);
            mxFree(meta);
            mexErrMsgIdAndTxt("MATLAB:hmtimes:dimMismatch",
                "hmtimes ==> Inner matrix dimensions must agree.");
        }
        
        // load data
        int numelX = (int)mxGetNumberOfElements(prhs[1]);
        cmpx *x = (cmpx*)mxMalloc(sizeof(cmpx) * numelX);
        double2complex(x, mxGetPr(prhs[1]), mxGetPi(prhs[1]), numelX);
        
       // do multiplication
        int ldy = 0;
        int nCols = (int)mxGetN(prhs[1]);
        cmpx *y = h->RMultDense(x, nCols, (int)mxGetM(prhs[1]), ldy);
        
        // store result
		plhs[0] = mxCreateDoubleMatrix(hmatDims[0], nCols, mxCOMPLEX);
		double *ansPr = mxGetPr(plhs[0]), *ansPi = mxGetPi(plhs[0]);
		for (int i = 0; i < nCols; i++)
			complex2double(ansPr + i*hmatDims[0], ansPi + i*hmatDims[0], y + i*ldy, hmatDims[0]);

        // clear
        mxFree(y);
        mxFree(x);
        delete h;
        mxFree(data);
        mxFree(meta);
    }
    else
    {
        // read in struct data
        double *data = NULL;
        readStructData(prhs[0], &data, &meta, dataLen, metaLen);
        
        // make a hierarchical matrix
        HMat<double> *h = new HMat<double>(data, meta, dataLen, metaLen);
        
        // check dimensions
        int hmatDims[2];
        h->Dims(hmatDims);
        if (hmatDims[1] != (int)mxGetM(prhs[1]))
        {
            delete h;
            mxFree(data);
            mxFree(meta);
            mexErrMsgIdAndTxt("MATLAB:hmtimes:dimMismatch",
                "hmtimes ==> Inner matrix dimensions must agree.");
        }
        
        // load data
        int numelX = (int)mxGetNumberOfElements(prhs[1]);
        double *x = (double*)mxMalloc(sizeof(double) * numelX);
        memcpy(x, mxGetPr(prhs[1]), sizeof(double) * numelX);
        
        // do multiplication
        int ldy = 0;
        int nCols = (int)mxGetN(prhs[1]);
        double *y = h->RMultDense(x, nCols, (int)mxGetM(prhs[1]), ldy);
        
        // store result
		plhs[0] = mxCreateDoubleMatrix(hmatDims[0], nCols, mxREAL);
		double *ansPr = mxGetPr(plhs[0]);
		for (int i = 0; i < nCols; i++)
			memcpy(ansPr + i*hmatDims[0], y + i*ldy, sizeof(double)*hmatDims[0]);
        
        // clear
        mxFree(y);
        mxFree(x);
        delete h;
        mxFree(data);
        mxFree(meta);
    }
    
}
