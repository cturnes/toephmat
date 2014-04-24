/*
    @(#)File:                /hminv.cpp
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        20 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Hierarchical matrix inversion
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
    if (nrhs != 1)
        mexErrMsgIdAndTxt("MATLAB:hminv:nrhs",
            "hminv ==> Incorrect number of inputs;\nSyntax is G = hminv(H); (see help).");
    if (mxIsStruct(prhs[0]) == false)
        mexErrMsgIdAndTxt("MATLAB:hminv:noStructData",
            "hminv ==> Hierarchical matrix must be supplied as a struct (see help for dense2hm).");
    int dataFieldNumber = (int)mxGetFieldNumber(prhs[0], "data");
    int metaFieldNumber = (int)mxGetFieldNumber(prhs[0], "meta");
    if ((dataFieldNumber == -1) || (metaFieldNumber == -1))
        mexErrMsgIdAndTxt("MATLAB:hmtimes:noStructData",
            "hminv ==> Struct data for hierarchical matrix is missing one or both fields.");
        
    // check whether matrix is complex
    bool isComplex = IsHMatComplex(prhs[0]);
    
    int *meta = NULL, dataLen = 0, metaLen = 0;
    if (isComplex == true)
    {
        // read in struct data
        cmpx *data = NULL;
        readStructData(prhs[0], &data, &meta, dataLen, metaLen);
        
        // make a hierarchical matrix
        HMat<cmpx> *h = new HMat<cmpx>(data, meta, dataLen, metaLen);
        mxFree(data);
        mxFree(meta);
        data = NULL;
        meta = NULL;
        dataLen = 0;
        metaLen = 0;
        
        // compute inverse
        HMat<cmpx> *hi = h->Invert();
        delete h;
        hi->GenerateOutput(&data, &meta, dataLen, metaLen);
        delete hi;
        
        // create output
        plhs[0] = writeStructData(data, meta, dataLen, metaLen);
        
        // clean up
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
        mxFree(data);
        mxFree(meta);
        data = NULL;
        meta = NULL;
        dataLen = 0;
        metaLen = 0;
        
        // compute inverse
        HMat<double> *hi = h->Invert();
        delete h;
        hi->GenerateOutput(&data, &meta, dataLen, metaLen);
        delete hi;
        
        // create output
        plhs[0] = writeStructData(data, meta, dataLen, metaLen);
        
        // clean up
        mxFree(data);
        mxFree(meta);
    }
    
}
