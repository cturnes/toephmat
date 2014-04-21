/*
    @(#)File:                /dense2hm.cpp
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        20 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Convert a dense matrix to a hierarchical matrix
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
    if ((nrhs < 1) || (nrhs > 2))
        mexErrMsgIdAndTxt("MATLAB:dense2hm:nrhs",
            "dense2hm ==> Incorrect dense2hm of inputs;\nSyntax is H = dense2hm(A); or H = dense2hm(A, LIM); (see help).");
    if (mxIsEmpty(prhs[0]) == true)
        mexErrMsgIdAndTxt("MATLAB:dense2hm:emptyInput",
            "dense2hm ==> Supplied matrix A cannot be empty.");
        
    // check whether scalar parameter is ok
    int nlim = 64;
    if ((nrhs > 1) && (mxIsEmpty(prhs[1]) == false))
    {
        double *npr = mxGetPr(prhs[1]);
        if ((int)npr[0] < 1)
            mexErrMsgIdAndTxt("MATLAB:dense2hm:badLimit",
            "dense2hm ==> Supplied limit value is non-positive.");
        else
            nlim = (int)npr[0];
    }
    
    // matrix dimensions
    int m = (int)mxGetM(prhs[0]), n = (int)mxGetN(prhs[0]);
    
    // check whether matrix is complex
    bool isComplex = mxIsComplex(prhs[0]);
    int *meta = NULL, dataLen = 0, metaLen = 0;
    if (isComplex == true)
    {
        // create hierarchical matrix
        cmpx *inData = (cmpx*)mxMalloc(sizeof(cmpx) * m * n);
        double2complex(inData, mxGetPr(prhs[0]), mxGetPi(prhs[0]), m*n);
        HMat<cmpx> *h = new HMat<cmpx>(inData, m, m, n, nlim);
        mxFree(inData);
        
        
        // generate hierarchical matrix storage
        cmpx *dataOut = NULL;
        h->GenerateOutput(&dataOut, &meta, dataLen, metaLen);
        delete h;
        
        // create output
        if (nlhs > 0)
            plhs[0] = writeStructData(dataOut, meta, dataLen, metaLen);
        
        // clean up
        mxFree(dataOut);
        mxFree(meta);
    }
    else
    {
        // create hierarchical matrix
        HMat<double> *h = new HMat<double>(mxGetPr(prhs[0]), m, m, n, nlim);
        
        // generate hierarchical matrix storage
        double *dataOut = NULL;
        h->GenerateOutput(&dataOut, &meta, dataLen, metaLen);
        delete h;
        
        // create output
        if (nlhs > 0)
            plhs[0] = writeStructData(dataOut, meta, dataLen, metaLen);
        
        // clean up
        mxFree(dataOut);
        mxFree(meta);
    }
}