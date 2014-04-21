/*
    @(#)File:                /kernel/misc.cpp
    @(#)Product:             Hierarchical Two-level Toeplitz Inversion
    @(#)Version:             1.0.0
    @(#)Last changed:        20 April 2014
    @(#)Author:              C. Turnes
    @(#)Copyright:           Georgia Institute of Technology
    @(#)Purpose:             Defines miscellaneous routines
*/
#include "misc.h"

// convert a double array to a complex array
void double2complex(cmpx *dest, double *sourceReal, double *sourceImag, const int n)
{
    if ((dest == NULL) || (sourceReal == NULL))
    {
        mexErrMsgIdAndTxt("MATLAB:misc:double2complex:badInput",
                          "One or more input arrays are NULL.");
    }
    if (sourceImag == NULL)
    {
        for (int i = 0; i < n; i++)
            dest[i] = cmpx(sourceReal[i], 0.0);
    }
    else
    {
        for (int i = 0; i < n; i++)
            dest[i] = cmpx(sourceReal[i], sourceImag[i]);
    }
}

// convert a double array to a complex array
void complex2double(double *destReal, double *destImag, cmpx *source, const int n)
{
    if ((destReal == NULL) || (destImag == NULL) || (source == NULL))
    {
        mexErrMsgIdAndTxt("MATLAB:misc:complex2double:badInput",
                          "One or more input arrays are NULL.");
    }

    for (int i = 0; i < n; i++)
    {
        destReal[i] = real(source[i]);
        destImag[i] = imag(source[i]);
    }
}

// integer maximum
int imin(const int a, const int b) { return (a < b) ? a : b; }

// integer minimum
int imax(const int a, const int b) { return (a > b) ? a : b; }

// compression of low-rank data - double matrices
void Compress(double** inA, int lda, int m,
              double** inB, int ldb, int n,
              int& r)
{
    // input validation
    if ((inA == NULL) || (inB == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
            "One or more essential input arrays are NULL.");
    double *a = (*inA);
    double *b = (*inB);
    if ((a == NULL) || (b == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
            "One or more essential input arrays are NULL.");
    if ((lda < m) || (ldb < n))
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
            "One or more supplied leading dimensions are smaller than the matrix side lengths.");
    if (r < 0)
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
                          "Invalid rank supplied.");
    
    // ensure low-rank product is actually low-rank
    if (r < imin(m, n))
    {
        /* ---- step 1: QR factorization of a ---- */
        // allocate for pivot
        int *jpvt = (int*)mxMalloc(sizeof(int) * 3 * r);
        memset(jpvt, 0, sizeof(int) * 3 * r);
        double *tau = (double*)mxMalloc(sizeof(double) * 2 * r);
        int lwork = -1, info;
        double *work = NULL, workopt;
        
        // call lapack QR factorization to query lwork
        dgeqp3(&m, &r, a, &lda, jpvt, tau, &workopt, &lwork, &info);
        lwork = (int)workopt;
        work = (double*)mxMalloc(sizeof(double)*lwork);
        dgeqp3(&m, &r, a, &lda, jpvt, tau, work, &lwork, &info);
        
        /* ---- step 2: QR factorization of b ---- */
        dgeqp3(&n, &r, b, &ldb, jpvt + r, tau + r, work, &lwork, &info);
        
        /* ---- step 3: Multiplication of R1*P1'*P2*R2' ---- */
        double *temp = (double*)mxMalloc(sizeof(double)*r*r*2);
        memset(temp, 0, sizeof(double)*r*r*2);
        // determine permutation of columns of R1
        for (int i = 0; i < r; i++)
        {
            int thisIdx = jpvt[i + r];
            for (int j = 0; j < r; j++)
            {
                if (jpvt[j] == thisIdx)
                {
                    jpvt[2*r + i] = j;
                    break;
                }
            }
        }
        // copy R1, permuting the order of columns
        for (int i = 0; i < r; i++)
        {
            int colIdx = jpvt[2*r + i];
            memcpy(temp + i*r, a + colIdx * lda, sizeof(double) * (colIdx + 1));
        }
        mxFree(jpvt);
        char side = 'R', uplo = 'U', transA = 'C', diag = 'N';
        double alpha = 1.0;
        dtrmm(&side, &uplo, &transA, &diag, &r, &r, &alpha, b, &ldb, temp, &r);
        
        /* ---- step 4: SVD of (R1*R2)' ---- */
        char jobu = 'O', jobvt = 'S';
        double *s = (double*)mxMalloc(sizeof(double) * r);
        int newLwork = -1;
        dgesvd(&jobu, &jobvt, &r, &r, temp, &r, s, temp, &r, temp + r*r, &r,
               &workopt, &newLwork, &info);
        newLwork = (int)workopt;
        if (newLwork != lwork)
        {
            work = (double*)mxRealloc(work, sizeof(double) * newLwork);
            lwork = newLwork;
        }
        dgesvd(&jobu, &jobvt, &r, &r, temp, &r, s, temp, &r, temp + r*r, &r,
               work, &newLwork, &info);
        
        /* ---- step 5: Determine new rank ---- */
        double tol = s[0] * imin(m, n) * EPS_VAL;
        int nsv = r;
        for (int i = (r-1); i >= 0; i--)
        {
            if (s[i] >= tol)
                break;
            else
                nsv--;
        }
        
        /* ---- step 6: (Q1)*U*S ---- */
        // first compute U*S for the vectors we care about
        double *anew = (double*)mxMalloc(sizeof(double) * lda * nsv);
        memset(anew, 0, sizeof(double) * lda * nsv);
        for (int i = 0; i < nsv; i++)
        {
            for (int j = 0; j < r; j++)
                anew[i*lda + j] = temp[i*r + j] * sqrt(s[i]);
        }
        side = 'L';
        transA = 'N';
        newLwork = -1;
        dormqr(&side, &transA, &m, &nsv, &r, a, &lda, tau, anew, &lda,
               &workopt, &newLwork, &info);
        newLwork = (int)workopt;
        if (newLwork != lwork)
        {
            work = (double*)mxRealloc(work, sizeof(double) * newLwork);
            lwork = newLwork;
        }
        dormqr(&side, &transA, &m, &nsv, &r, a, &lda, tau, anew, &lda, work,
               &newLwork, &info);
        mxFree(a);
        (*inA) = anew;
        
        /* ---- step 7: V'*Q2' ---- */
        // apply the orthogonal transform Q2
        double *bnew = (double*)mxMalloc(sizeof(double) * ldb * nsv);
        memset(bnew, 0, sizeof(double) * ldb * nsv);
        for (int i = 0; i < nsv; i++)
        {
            for (int j = 0; j < r; j++)
                bnew[i*ldb + j] = temp[r*r + j*r + i] * sqrt(s[i]);
        }
        mxFree(s);
        newLwork = -1;
        dormqr(&side, &transA, &n, &nsv, &r, b, &ldb, tau + r, bnew, &ldb, &workopt, &newLwork, &info);
        newLwork = (int)workopt;
        if (newLwork != lwork)
        {
            work = (double*)mxRealloc(work, sizeof(double) * newLwork);
            lwork = newLwork;
        }
        dormqr(&side, &transA, &n, &nsv, &r, b, &ldb, tau + r, bnew, &ldb, work, &newLwork, &info);
        mxFree(b);
        (*inB) = bnew;
        
        /* ---- step 8: Update the rank ---- */
        r = nsv;
        
        /* ---- step 9: Cleanup ---- */
        mxFree(temp);
        mxFree(work);
        mxFree(tau);
    }
    else
    {
        // compute the product of the two sides
        char transA = 'N', transB = 'C';
        double *C = (double*)mxMalloc(sizeof(double) * m * n);
        double alpha = 1.0, beta = 0.0;
        dgemm(&transA, &transB, &m, &n, &r, &alpha, a, &lda, b, &ldb,
              &beta, C, &m);
        
        // take the SVD
        char jobu = 'O', jobvt = 'S';
        int newRank = imin(m, n);
        double *s = (double*)mxMalloc(sizeof(double) * newRank);
        double *temp = (double*)mxMalloc(sizeof(double) * newRank * n);
        int lwork = -1, info;
        double *work = NULL, workopt;
        dgesvd(&jobu, &jobvt, &m, &n, C, &m, s, C, &m, temp, &newRank,
               &workopt, &lwork, &info);
        lwork = (int)workopt;
        work = (double*)mxRealloc(work, sizeof(double) * lwork);
        dgesvd(&jobu, &jobvt, &m, &n, C, &m, s, C, &m, temp, &newRank,
               work, &lwork, &info);
        mxFree(work);
        
        // determine new rank
        double tol = s[0] * newRank * EPS_VAL;
        int nsv = newRank;
        for (int i = (newRank-1); i >= 0; i--)
        {
            if (s[i] >= tol)
                break;
            else
                nsv--;
        }
        
        // reallocate
        mxFree(a);
        a = (double*)mxMalloc(sizeof(double) * nsv * m);
        mxFree(b);
        b = (double*)mxMalloc(sizeof(double) * nsv * n);
        for (int j = 0; j < nsv; j++)
        {
            for (int i = 0; i < m; i++)
                a[i + j*m] = C[i + j*m] * sqrt(s[j]);
            for (int i = 0; i < n; i++)
                b[i + j*n] = temp[j + i*newRank] * sqrt(s[j]);
        }
        mxFree(s);
        mxFree(temp);
        mxFree(C);
        (*inA) = a;
        (*inB) = b;
        r = nsv;
    }
}

// compression of low-rank data - complex matrices
void Compress(cmpx** inA, int lda, int m,
              cmpx** inB, int ldb, int n,
              int& r)
{
    // input validation
    if ((inA == NULL) || (inB == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
            "One or more essential input arrays are NULL.");
    cmpx *a = (*inA);
    cmpx *b = (*inB);
    if ((a == NULL) || (b == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
            "One or more essential input arrays are NULL.");
    if ((lda < m) || (ldb < n))
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
            "One or more supplied leading dimensions are smaller than the matrix side lengths.");
    if (r < 0)
        mexErrMsgIdAndTxt("MATLAB:misc:Compress:badInput",
                          "Invalid rank supplied.");
    
    // ensure low-rank product is actually low-rank
    if (r < imin(m, n))
    {
        /* ---- step 1: QR factorization of a ---- */
        // allocate for pivot
        int *jpvt = (int*)mxMalloc(sizeof(int) * 3 * r);
        memset(jpvt, 0, sizeof(int) * 3 * r);
        cmpx *tau = (cmpx*)mxMalloc(sizeof(cmpx) * 2 * r);
        int lwork = -1, info;
        cmpx *work = NULL, workopt;
        
        // call lapack QR factorization to query lwork
        double *rwork = (double*)mxMalloc(sizeof(double) * 5 * r);
        zgeqp3(&m, &r, CCAST(a), &lda, jpvt, CCAST(tau), CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxMalloc(sizeof(cmpx) * lwork);
        zgeqp3(&m, &r, CCAST(a), &lda, jpvt, CCAST(tau), CCAST(work), &lwork, rwork, &info);
        
        /* ---- step 2: QR factorization of b ---- */
        zgeqp3(&n, &r, CCAST(b), &ldb, jpvt + r, CCAST(tau + r), CCAST(work), &lwork, rwork, &info);
        
        /* ---- step 3: Multiplication of R1*P1'*P2*R2' ---- */
        cmpx *temp = (cmpx*)mxMalloc(sizeof(cmpx)*r*r*2);
        memset(temp, 0, sizeof(cmpx)*r*r*2);
        // determine permutation of columns of R1
        for (int i = 0; i < r; i++)
        {
            int thisIdx = jpvt[i + r];
            for (int j = 0; j < r; j++)
            {
                if (jpvt[j] == thisIdx)
                {
                    jpvt[2*r + i] = j;
                    break;
                }
            }
        }
        // copy R1, permuting the order of columns
        for (int i = 0; i < r; i++)
        {
            int colIdx = jpvt[2*r + i];
            memcpy(temp + i*r, a + colIdx * lda, sizeof(cmpx) * (colIdx + 1));
        }
        mxFree(jpvt);
        char side = 'R', uplo = 'U', transA = 'C', diag = 'N';
        cmpx alpha = cmpx(1.0, 0.0);
        ztrmm(&side, &uplo, &transA, &diag, &r, &r, CCAST(&alpha), CCAST(b), &ldb, CCAST(temp), &r);
        
        /* ---- step 4: SVD of (R1*R2)' ---- */
        char jobu = 'O', jobvt = 'S';
        double *s = (double*)mxMalloc(sizeof(double) * r);
        int newLwork = -1;
        zgesvd(&jobu, &jobvt, &r, &r, CCAST(temp), &r, s, CCAST(temp), &r, CCAST(temp + r*r), &r,
               CCAST(&workopt), &newLwork, rwork, &info);
        newLwork = (int)real(workopt);
        if (newLwork != lwork)
        {
            work = (cmpx*)mxRealloc(work, sizeof(cmpx) * newLwork);
            lwork = newLwork;
        }
        zgesvd(&jobu, &jobvt, &r, &r, CCAST(temp), &r, s, CCAST(temp), &r, CCAST(temp + r*r), &r,
               CCAST(work), &newLwork, rwork, &info);
        
        /* ---- step 5: Determine new rank ---- */
        double tol = s[0] * imin(m, n) * EPS_VAL;
        int nsv = r;
        for (int i = (r-1); i >= 0; i--)
        {
            if (s[i] >= tol)
                break;
            else
                nsv--;
        }
        
        /* ---- step 6: (Q1)*U*S ---- */
        // first compute U*S for the vectors we care about
        cmpx *anew = (cmpx*)mxMalloc(sizeof(cmpx) * lda * nsv);
        memset(anew, 0, sizeof(cmpx) * lda * nsv);
        for (int i = 0; i < nsv; i++)
        {
            for (int j = 0; j < r; j++)
                anew[i*lda + j] = temp[i*r + j] * sqrt(s[i]);
        }
        side = 'L';
        transA = 'N';
        newLwork = -1;
        zunmqr(&side, &transA, &m, &nsv, &r, CCAST(a), &lda, CCAST(tau), CCAST(anew), &lda, CCAST(&workopt), &newLwork, &info);
        newLwork = (int)real(workopt);
        if (newLwork != lwork)
        {
            work = (cmpx*)mxRealloc(work, sizeof(cmpx) * newLwork);
            lwork = newLwork;
        }
        zunmqr(&side, &transA, &m, &nsv, &r, CCAST(a), &lda, CCAST(tau), CCAST(anew), &lda, CCAST(work), &newLwork, &info);
        mxFree(a);
        (*inA) = anew;
        
        /* ---- step 7: V'*Q2' ---- */
        // apply the orthogonal transform Q2
        cmpx *bnew = (cmpx*)mxMalloc(sizeof(cmpx) * ldb * nsv);
        memset(bnew, 0, sizeof(cmpx) * ldb * nsv);
        for (int i = 0; i < nsv; i++)
        {
            for (int j = 0; j < r; j++)
                bnew[i*ldb + j] = conj(temp[r*r + j*r + i]) * sqrt(s[i]);
        }
        mxFree(s);
        newLwork = -1;
        zunmqr(&side, &transA, &n, &nsv, &r, CCAST(b), &ldb, CCAST(tau + r), CCAST(bnew), &ldb, CCAST(&workopt), &newLwork, &info);
        newLwork = (int)real(workopt);
        if (newLwork != lwork)
        {
            work = (cmpx*)mxRealloc(work, sizeof(cmpx) * newLwork);
            lwork = newLwork;
        }
        zunmqr(&side, &transA, &n, &nsv, &r, CCAST(b), &ldb, CCAST(tau + r), CCAST(bnew), &ldb, CCAST(work), &newLwork, &info);
        mxFree(b);
        (*inB) = bnew;
        
        /* ---- step 8: Update the rank ---- */
        r = nsv;
        
        /* ---- step 9: Cleanup ---- */
        mxFree(temp);
        mxFree(work);
        mxFree(tau);
    }
    else
    {
        // compute the product of the two sides
        char transA = 'N', transB = 'C';
        cmpx *C = (cmpx*)mxMalloc(sizeof(cmpx) * m * n);
        cmpx alpha = 1.0, beta = 0.0;
        zgemm(&transA, &transB, &m, &n, &r, CCAST(&alpha), CCAST(a), &lda,
              CCAST(b), &ldb, CCAST(&beta), CCAST(C), &m);
        
        // take the SVD
        char jobu = 'O', jobvt = 'S';
        int newRank = imin(m, n);
        double *s = (double*)mxMalloc(sizeof(double) * newRank);
        cmpx *temp = (cmpx*)mxMalloc(sizeof(cmpx) * newRank * n);
        int lwork = -1, info;
        cmpx *work = NULL, workopt;
        double *rwork = (double*)mxMalloc(sizeof(double) * newRank * 5);
        zgesvd(&jobu, &jobvt, &m, &n, CCAST(C), &m, s, CCAST(C), &m, CCAST(temp), &newRank,
               CCAST(&workopt), &lwork, rwork, &info);
        lwork = (int)real(workopt);
        work = (cmpx*)mxRealloc(work, sizeof(cmpx) * lwork);
        zgesvd(&jobu, &jobvt, &m, &n, CCAST(C), &m, s, CCAST(C), &m, CCAST(temp), &newRank,
               CCAST(work), &lwork, rwork, &info);
        mxFree(work);
        mxFree(rwork);
        
        // determine new rank
        double tol = s[0] * newRank * EPS_VAL;
        int nsv = newRank;
        for (int i = (newRank-1); i >= 0; i--)
        {
            if (s[i] >= tol)
                break;
            else
                nsv--;
        }
        
        // reallocate
        mxFree(a);
        a = (cmpx*)mxMalloc(sizeof(cmpx) * nsv * m);
        mxFree(b);
        b = (cmpx*)mxMalloc(sizeof(cmpx) * nsv * n);
        for (int j = 0; j < nsv; j++)
        {
            for (int i = 0; i < m; i++)
                a[i + j*m] = C[i + j*m] * sqrt(s[j]);
            for (int i = 0; i < n; i++)
                b[i + j*n] = conj(temp[j + i*newRank]) * sqrt(s[j]);
        }
        mxFree(s);
        mxFree(temp);
        mxFree(C);
        (*inA) = a;
        (*inB) = b;
        r = nsv;
    }
}

// vector transpose in place
void TransposeIP(double** in, const int nRows, const int nCols, int& ldim)
{
    if (in == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeIP:badInput",
                          "Input matrix is NULL.");
    }
    if ((nRows < 1) || (nCols < 1))
    {
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeIP:badInput",
                          "Input matrix cannot be empty.");
    }
    if (ldim < nRows)
    {
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeIP:badInput",
                          "Supplied leading dimension for input matrix is invalid.");
    }
    
    // allocate for result
    double* out = (double*)mxMalloc(sizeof(double) * nCols * nRows);
    // input array
    double* input = (*in);
    for (int j = 0; j < nCols; j++)
    {
        for (int i = 0; i < nRows; i++)
            out[j + i*nCols] = input[i + j*ldim];
    }
    
    // adjust array
    mxFree(*in);
    (*in) = out;
    ldim = nCols;
}

// vector transpose in place
void TransposeIP(cmpx** in, const int nRows, const int nCols, int& ldim)
{
    if (in == NULL)
    {
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeIP:badInput",
                          "Input matrix is NULL.");
    }
    if ((nRows < 1) || (nCols < 1))
    {
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeIP:badInput",
                          "Input matrix cannot be empty.");
    }
    if (ldim < nRows)
    {
        mexErrMsgIdAndTxt("MATLAB:misc:TransposeIP:badInput",
                          "Supplied leading dimension for input matrix is invalid.");
    }
    
    // allocate for result
    cmpx* out = (cmpx*)mxMalloc(sizeof(cmpx) * nCols * nRows);
    // input array
    cmpx* input = (*in);
    for (int j = 0; j < nCols; j++)
    {
        for (int i = 0; i < nRows; i++)
            out[j + i*nCols] = conj(input[i + j*ldim]);
    }
    
    // adjust array
    mxFree(*in);
    (*in) = out;
    ldim = nCols;
}

// checks whether the data within a hierarchical structure is real or complex
bool IsHMatComplex(const mxArray* source)
{
    // input validation
    if (mxIsStruct(source) == false)
        mexErrMsgIdAndTxt("MATLAB:misc:IsHMatComplex:noStructData",
            "Hierarchical matrix must be supplied as a struct (see help for dense2hm).");
    int dataFieldNumber = (int)mxGetFieldNumber(source, "data");
    if (dataFieldNumber == -1)
        mexErrMsgIdAndTxt("MATLAB:misc:IsHMatComplex:noStructData",
            "Struct for hierarchical matrix is missing data field.");
    
    // get pointer to array
    mxArray *dataArray = mxGetFieldByNumber(source, 0, dataFieldNumber);
    
    return mxIsComplex(dataArray);
}

// reads structure data
void readStructData(const mxArray *source, cmpx **data, int **meta, int& dataLen, int& metaLen)
{
    if ((data == NULL) || (meta == NULL) || (source == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:badInput",
            "One or more input arrays are NULL.");
        
    // input validation
    if (mxIsStruct(source) == false)
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:noStructData",
            "Hierarchical matrix must be supplied as a struct (see help for dense2hm).");
    int dataFieldNumber = (int)mxGetFieldNumber(source, "data");
    if (dataFieldNumber == -1)
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:noStructData",
            "Struct for hierarchical matrix is missing data field.");
    int metaFieldNumber = (int)mxGetFieldNumber(source, "meta");
    if (metaFieldNumber == -1)
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:noStructData",
            "Struct for hierarchical matrix is missing meta field.");
        
    // first load data
    mxArray *mxData = mxGetFieldByNumber(source, 0, dataFieldNumber);
    dataLen = (int)mxGetNumberOfElements(mxData);
    (*data) = (cmpx*)mxMalloc(sizeof(cmpx) * dataLen);
    double *prData = mxGetPr(mxData), *piData = mxGetPi(mxData);
    double2complex(*data, prData, piData, dataLen);
    
    // then load meta data
    mxArray *mxMeta = mxGetFieldByNumber(source, 0, metaFieldNumber);
    int numel = (int)mxGetNumberOfElements(mxMeta);
    (*meta) = (int*)mxMalloc(sizeof(int) * numel);
    memcpy(*meta, (int*)mxGetData(mxMeta), sizeof(int) * numel);
    metaLen = (int)mxGetN(mxMeta);
}

// reads structure data
void readStructData(const mxArray *source, double **data, int **meta, int& dataLen, int& metaLen)
{
    if ((data == NULL) || (meta == NULL) || (source == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:badInput",
            "One or more input arrays are NULL.");
    
    // input validation
    if (mxIsStruct(source) == false)
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:noStructData",
            "Hierarchical matrix must be supplied as a struct (see help for dense2hm).");
    int dataFieldNumber = (int)mxGetFieldNumber(source, "data");
    if (dataFieldNumber == -1)
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:noStructData",
            "Struct for hierarchical matrix is missing data field.");
    int metaFieldNumber = (int)mxGetFieldNumber(source, "meta");
    if (metaFieldNumber == -1)
        mexErrMsgIdAndTxt("MATLAB:misc:readStructData:noStructData",
            "Struct for hierarchical matrix is missing meta field.");
        
    // first load data
    mxArray *mxData = mxGetFieldByNumber(source, 0, dataFieldNumber);
    dataLen = (int)mxGetNumberOfElements(mxData);
    (*data) = (double*)mxMalloc(sizeof(double) * dataLen);
    memcpy(*data, mxGetPr(mxData), sizeof(double) * dataLen);
    
    // then load meta data
    mxArray *mxMeta = mxGetFieldByNumber(source, 0, metaFieldNumber);
    int numel = (int)mxGetNumberOfElements(mxMeta);
    (*meta) = (int*)mxMalloc(sizeof(int) * numel);
    memcpy(*meta, (int*)mxGetData(mxMeta), sizeof(int) * numel);
    metaLen = (int)mxGetN(mxMeta);
}

// writes structure data
mxArray* writeStructData(double *data, int *meta, const int dataLen, const int metaLen)
{
    // input validation
    if ((data == NULL) || (meta == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:writeStructData:badInput",
            "One or more input arrays are NULL.");
    if ((dataLen < 0) || (metaLen < 0))
        mexErrMsgIdAndTxt("MATLAB:misc:writeStructData:badInput",
            "One or more supplied array lengths are invalid.");
        
    // create structure
    const char *keys[] = { "data", "meta" };
    mxArray* output = mxCreateStructMatrix(1, 1, 2, keys);
    
    // data array
    mxArray* mxData = mxCreateDoubleMatrix(dataLen, 1, mxREAL);
    double *dataPr = mxGetPr(mxData);
    memcpy(dataPr, data, sizeof(double) * dataLen);
    mxSetField(output, 0, keys[0], mxData);
    
    // metadata array
    mxArray* mxMeta = mxCreateNumericMatrix((mwSize)NUMMETA, (mwSize)metaLen, mxINT32_CLASS, mxREAL);
    int *metaPr = (int*)mxGetData(mxMeta);
    memcpy(metaPr, meta, sizeof(int) * NUMMETA * metaLen);
    mxSetField(output, 0, keys[1], mxMeta);
    
    return output;
}

// writes structure data
mxArray* writeStructData(cmpx *data, int *meta, const int dataLen, const int metaLen)
{
    // input validation
    if ((data == NULL) || (meta == NULL))
        mexErrMsgIdAndTxt("MATLAB:misc:writeStructData:badInput",
            "One or more input arrays are NULL.");
    if ((dataLen < 0) || (metaLen < 0))
        mexErrMsgIdAndTxt("MATLAB:misc:writeStructData:badInput",
            "One or more supplied array lengths are invalid.");
        
    // create structure
    const char *keys[] = { "data", "meta" };
    mxArray* output = mxCreateStructMatrix(1, 1, 2, keys);
    
    // data array
    mxArray* mxData = mxCreateDoubleMatrix(dataLen, 1, mxCOMPLEX);
    double *dataPr = mxGetPr(mxData), *dataPi = mxGetPi(mxData);
    complex2double(dataPr, dataPi, data, dataLen);
    mxSetField(output, 0, keys[0], mxData);
    
    // metadata array
    mxArray* mxMeta = mxCreateNumericMatrix((mwSize)NUMMETA, (mwSize)metaLen, mxINT32_CLASS, mxREAL);
    int *metaPr = (int*)mxGetData(mxMeta);
    memcpy(metaPr, meta, sizeof(int) * NUMMETA * metaLen);
    mxSetField(output, 0, keys[1], mxMeta);
    
    return output;
}