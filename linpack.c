/*
**
** LINPACK.C        Linpack benchmark, calculates FLOPS.
**                  (FLoating Point Operations Per Second)
**
** Translated to C by Bonnie Toy 5/88
**
** Modified by Will Menninger, 10/93, with these features:
**  (modified on 2/25/94  to fix a problem with daxpy  for
**   unequal increments or equal increments not equal to 1.
**     Jack Dongarra)
**
** - Defaults to double precision.
** - Averages ROLLed and UNROLLed performance.
** - User selectable array sizes.
** - Automatically does enough repetitions to take at least 10 CPU seconds.
** - Prints machine precision.
** - ANSI prototyping.
**
** To compile:  cc -O -o linpack linpack.c -lm
**
**
*/

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <float.h>
#include <errno.h>
#include <limits.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>

#define DP

#ifdef SP
#define ZERO        0.0
#define ONE         1.0
#define PREC        "Single"
#define BASE10DIG   FLT_DIG

typedef float   REAL;
#endif

#ifdef DP
#define ZERO        0.0e0
#define ONE         1.0e0
#define PREC        "Double"
#define BASE10DIG   DBL_DIG

typedef double  REAL;
#endif

typedef struct {
    int arsize;
    long nreps;
    void *mempool;
    REAL tdgesl;
    REAL tdgefa;
} worker_data;

static REAL linpack  (long nreps,int nthreads,worker_data *workers,int *status);
static void *linpack_worker(void *arg);
static int available_cpus(void);
static void matgen   (REAL *a,int lda,int n,REAL *b,REAL *norma);
static void dgefa    (REAL *a,int lda,int n,int *ipvt,int *info,int roll);
static void dgesl    (REAL *a,int lda,int n,int *ipvt,REAL *b,int job,int roll);
static void daxpy_r  (int n,REAL da,REAL *dx,int incx,REAL *dy,int incy);
static REAL ddot_r   (int n,REAL *dx,int incx,REAL *dy,int incy);
static void dscal_r  (int n,REAL da,REAL *dx,int incx);
static void daxpy_ur (int n,REAL da,REAL *dx,int incx,REAL *dy,int incy);
static REAL ddot_ur  (int n,REAL *dx,int incx,REAL *dy,int incy);
static void dscal_ur (int n,REAL da,REAL *dx,int incx);
static int  idamax   (int n,REAL *dx,int incx);
static REAL second   (void);

int main(void)
{
    char    *arsize_input;
    char    *threads_input;
    char    *end;
    int     arsize,nthreads,i,status;
    long    parsed_value;
    long    arsize2d,nreps;
    size_t  memreq,totalmem;
    REAL    elapsed;
    worker_data *workers;

    arsize_input = getenv("LINPACK_ARRAY_SIZE");
    if (arsize_input == NULL) {
        arsize = 200;
    } else {
        errno = 0;
        parsed_value = strtol(arsize_input,&end,10);
        if (errno != 0 || end == arsize_input || *end != '\0' ||
                parsed_value < 10 || parsed_value > INT_MAX - 1) {
            fprintf(stderr,"Invalid LINPACK_ARRAY_SIZE value.\n");
            return 1;
        }
        arsize = (int)parsed_value;
    }

    arsize/=2;
    arsize*=2;
    if (arsize<10) {
        printf("Too small.\n");
        return 1;
    }

    nthreads = available_cpus();
    threads_input = getenv("LINPACK_THREADS");
    if (threads_input != NULL) {
        errno = 0;
        parsed_value = strtol(threads_input,&end,10);
        if (errno != 0 || end == threads_input || *end != '\0' ||
                parsed_value < 1 || parsed_value > INT_MAX) {
            fprintf(stderr,"Invalid LINPACK_THREADS value.\n");
            return 1;
        }
        nthreads = (int)parsed_value;
    }

    if ((size_t)arsize > SIZE_MAX/(size_t)arsize/sizeof(REAL)) {
        fprintf(stderr,"Requested array size is too large.\n");
        return 2;
    }
    arsize2d = (long)arsize*(long)arsize;
    memreq=(size_t)arsize2d*sizeof(REAL);
    if (memreq > SIZE_MAX-(size_t)arsize*sizeof(REAL)-
            (size_t)arsize*sizeof(int)) {
        fprintf(stderr,"Requested array size is too large.\n");
        return 2;
    }
    memreq+=(size_t)arsize*sizeof(REAL)+(size_t)arsize*sizeof(int);
    if (memreq > SIZE_MAX / (size_t)nthreads) {
        fprintf(stderr,"Requested thread count requires too much memory.\n");
        return 2;
    }
    totalmem=memreq*(size_t)nthreads;
    printf("Memory required:  %zuK (%zuK per thread).\n",
            (totalmem>>10)+((totalmem&1023U)>=512U),
            (memreq>>10)+((memreq&1023U)>=512U));
    workers=calloc((size_t)nthreads,sizeof(*workers));
    if (workers==NULL) {
        fprintf(stderr,"Not enough memory available for worker data.\n\n");
        return 2;
    }
    for (i=0;i<nthreads;i++) {
        workers[i].arsize=arsize;
        workers[i].mempool=malloc(memreq);
        if (workers[i].mempool==NULL) {
            fprintf(stderr,"Not enough memory available for worker %d.\n\n",i);
            while (i > 0)
                free(workers[--i].mempool);
            free(workers);
            return 2;
        }
    }

    printf("\n\nLINPACK benchmark, %s precision.\n",PREC);
    printf("Machine precision:  %d digits.\n",BASE10DIG);
    printf("Array size %d X %d.\n",arsize,arsize);
    printf("Threads: %d (independent calculations).\n",nthreads);
    printf("Aggregate rolled and unrolled performance:\n\n");
    printf("    Reps Threads Time(s) DGEFA   DGESL  OVERHEAD    KFLOPS\n");
    printf("------------------------------------------------------------\n");
    nreps=1;
    do {
        status=0;
        elapsed=linpack(nreps,nthreads,workers,&status);
        if (elapsed<0.) {
            for (i=0;i<nthreads;i++)
                free(workers[i].mempool);
            free(workers);
            return status;
        }
        if (elapsed >= 10.)
            break;
        if (nreps > LONG_MAX / 2) {
            fprintf(stderr,"Unable to reach benchmark duration safely.\n");
            for (i=0;i<nthreads;i++)
                free(workers[i].mempool);
            free(workers);
            return 2;
        }
        nreps*=2;
    } while (1);

    for (i=0;i<nthreads;i++)
        free(workers[i].mempool);
    free(workers);
    printf("\n");
    return 0;
}


static REAL linpack(long nreps,int nthreads,worker_data *workers,int *status)
{
    REAL tdgesl,tdgefa,totalt,toverhead,ops,kflops;
    pthread_t *threads;
    int i,created,error;
    long n;
    struct timespec start,finish;

    if ((size_t)nthreads > SIZE_MAX/sizeof(*threads)) {
        fprintf(stderr,"Requested thread count is too large.\n");
        *status=2;
        return -1.;
    }
    threads=malloc((size_t)nthreads*sizeof(*threads));
    if (threads==NULL) {
        fprintf(stderr,"Unable to allocate thread handles.\n");
        *status=2;
        return -1.;
    }
    if (clock_gettime(CLOCK_MONOTONIC,&start)!=0) {
        fprintf(stderr,"Unable to read monotonic clock: %s\n",strerror(errno));
        free(threads);
        *status=2;
        return -1.;
    }
    for (i=0;i<nthreads;i++) {
        workers[i].nreps=nreps;
        error=pthread_create(&threads[i],NULL,linpack_worker,&workers[i]);
        if (error!=0) {
            fprintf(stderr,"Unable to create worker thread %d: %s\n",
                    i,strerror(error));
            *status=2;
            break;
        }
    }
    created=i;
    for (i=0;i<created;i++) {
        error=pthread_join(threads[i],NULL);
        if (error!=0) {
            fprintf(stderr,"Unable to join worker thread %d: %s\n",
                    i,strerror(error));
            *status=2;
        }
    }
    if (*status!=0) {
        free(threads);
        return -1.;
    }
    if (clock_gettime(CLOCK_MONOTONIC,&finish)!=0) {
        fprintf(stderr,"Unable to read monotonic clock: %s\n",strerror(errno));
        free(threads);
        *status=2;
        return -1.;
    }
    totalt=(REAL)(finish.tv_sec-start.tv_sec)+
            (REAL)(finish.tv_nsec-start.tv_nsec)/1.0e9;
    tdgesl=0.;
    tdgefa=0.;
    for (i=0;i<nthreads;i++) {
        tdgesl+=workers[i].tdgesl;
        tdgefa+=workers[i].tdgefa;
    }
    tdgesl/=nthreads;
    tdgefa/=nthreads;
    toverhead=totalt-tdgesl-tdgefa;
    if (toverhead<0.)
        toverhead=0.;
    n=workers[0].arsize/2;
    ops=((2.0*n*n*n)/3.0+2.0*n*n);
    kflops=(REAL)nthreads*2.*nreps*ops/(1000.*totalt);
    if (totalt>=0.5 && tdgefa+tdgesl>=0.2) {
        printf("%8ld %7d %6.2f %6.2f%% %6.2f%% %6.2f%%  %9.3f\n",
                nreps,nthreads,totalt,100.*tdgefa/totalt,
                100.*tdgesl/totalt,100.*toverhead/totalt,kflops);
    }
    free(threads);
    return(totalt);
}

static void *linpack_worker(void *arg)
{
    worker_data *worker=(worker_data *)arg;
    REAL *a,*b,norma,t1;
    int *ipvt,n,info,lda;
    long i,arsize2d;

    lda=worker->arsize;
    n=worker->arsize/2;
    arsize2d=(long)worker->arsize*(long)worker->arsize;
    a=(REAL *)worker->mempool;
    b=a+arsize2d;
    ipvt=(int *)&b[worker->arsize];
    worker->tdgesl=0.;
    worker->tdgefa=0.;
    for (i=0;i<worker->nreps;i++) {
        matgen(a,lda,n,b,&norma);
        t1=second();
        dgefa(a,lda,n,ipvt,&info,1);
        worker->tdgefa+=second()-t1;
        t1=second();
        dgesl(a,lda,n,ipvt,b,0,1);
        worker->tdgesl+=second()-t1;
    }
    for (i=0;i<worker->nreps;i++) {
        matgen(a,lda,n,b,&norma);
        t1=second();
        dgefa(a,lda,n,ipvt,&info,0);
        worker->tdgefa+=second()-t1;
        t1=second();
        dgesl(a,lda,n,ipvt,b,0,0);
        worker->tdgesl+=second()-t1;
    }
    return NULL;
}

static int available_cpus(void)
{
    long count;

#ifdef __linux__
    cpu_set_t cpus;
    if (sched_getaffinity(0,sizeof(cpus),&cpus)==0) {
        count=CPU_COUNT(&cpus);
        if (count>0 && count<=INT_MAX)
            return (int)count;
    }
#endif
    count=sysconf(_SC_NPROCESSORS_ONLN);
    if (count<1 || count>INT_MAX)
        return 1;
    return (int)count;
}


/*
** For matgen,
** We would like to declare a[][lda], but c does not allow it.  In this
** function, references to a[i][j] are written a[lda*i+j].
*/
static void matgen(REAL *a,int lda,int n,REAL *b,REAL *norma)

    {
    int init,i,j;

    init = 1325;
    *norma = 0.0;
    for (j = 0; j < n; j++)
        for (i = 0; i < n; i++)
            {
            init = (int)((long)3125*(long)init % 65536L);
            a[lda*j+i] = (init - 32768.0)/16384.0;
            *norma = (a[lda*j+i] > *norma) ? a[lda*j+i] : *norma;
            }
    for (i = 0; i < n; i++)
        b[i] = 0.0;
    for (j = 0; j < n; j++)
        for (i = 0; i < n; i++)
            b[i] = b[i] + a[lda*j+i];
    }


/*
**
** DGEFA benchmark
**
** We would like to declare a[][lda], but c does not allow it.  In this
** function, references to a[i][j] are written a[lda*i+j].
**
**   dgefa factors a double precision matrix by gaussian elimination.
**
**   dgefa is usually called by dgeco, but it can be called
**   directly with a saving in time if  rcond  is not needed.
**   (time for dgeco) = (1 + 9/n)*(time for dgefa) .
**
**   on entry
**
**      a       REAL precision[n][lda]
**              the matrix to be factored.
**
**      lda     integer
**              the leading dimension of the array  a .
**
**      n       integer
**              the order of the matrix  a .
**
**   on return
**
**      a       an upper triangular matrix and the multipliers
**              which were used to obtain it.
**              the factorization can be written  a = l*u  where
**              l  is a product of permutation and unit lower
**              triangular matrices and  u  is upper triangular.
**
**      ipvt    integer[n]
**              an integer vector of pivot indices.
**
**      info    integer
**              = 0  normal value.
**              = k  if  u[k][k] .eq. 0.0 .  this is not an error
**                   condition for this subroutine, but it does
**                   indicate that dgesl or dgedi will divide by zero
**                   if called.  use  rcond  in dgeco for a reliable
**                   indication of singularity.
**
**   linpack. this version dated 08/14/78 .
**   cleve moler, university of New Mexico, argonne national lab.
**
**   functions
**
**   blas daxpy,dscal,idamax
**
*/
static void dgefa(REAL *a,int lda,int n,int *ipvt,int *info,int roll)

    {
    REAL t;
    int idamax(),j,k,kp1,l,nm1;

    /* gaussian elimination with partial pivoting */

    if (roll)
        {
        *info = 0;
        nm1 = n - 1;
        if (nm1 >=  0)
            for (k = 0; k < nm1; k++)
                {
                kp1 = k + 1;

                /* find l = pivot index */

                l = idamax(n-k,&a[lda*k+k],1) + k;
                ipvt[k] = l;

                /* zero pivot implies this column already
                   triangularized */

                if (a[lda*k+l] != ZERO)
                    {

                    /* interchange if necessary */

                    if (l != k)
                        {
                        t = a[lda*k+l];
                        a[lda*k+l] = a[lda*k+k];
                        a[lda*k+k] = t;
                        }

                    /* compute multipliers */

                    t = -ONE/a[lda*k+k];
                    dscal_r(n-(k+1),t,&a[lda*k+k+1],1);

                    /* row elimination with column indexing */

                    for (j = kp1; j < n; j++)
                        {
                        t = a[lda*j+l];
                        if (l != k)
                            {
                            a[lda*j+l] = a[lda*j+k];
                            a[lda*j+k] = t;
                            }
                        daxpy_r(n-(k+1),t,&a[lda*k+k+1],1,&a[lda*j+k+1],1);
                        }
                    }
                else
                    (*info) = k;
                }
        ipvt[n-1] = n-1;
        if (a[lda*(n-1)+(n-1)] == ZERO)
            (*info) = n-1;
        }
    else
        {
        *info = 0;
        nm1 = n - 1;
        if (nm1 >=  0)
            for (k = 0; k < nm1; k++)
                {
                kp1 = k + 1;

                /* find l = pivot index */

                l = idamax(n-k,&a[lda*k+k],1) + k;
                ipvt[k] = l;

                /* zero pivot implies this column already
                   triangularized */

                if (a[lda*k+l] != ZERO)
                    {

                    /* interchange if necessary */

                    if (l != k)
                        {
                        t = a[lda*k+l];
                        a[lda*k+l] = a[lda*k+k];
                        a[lda*k+k] = t;
                        }

                    /* compute multipliers */

                    t = -ONE/a[lda*k+k];
                    dscal_ur(n-(k+1),t,&a[lda*k+k+1],1);

                    /* row elimination with column indexing */

                    for (j = kp1; j < n; j++)
                        {
                        t = a[lda*j+l];
                        if (l != k)
                            {
                            a[lda*j+l] = a[lda*j+k];
                            a[lda*j+k] = t;
                            }
                        daxpy_ur(n-(k+1),t,&a[lda*k+k+1],1,&a[lda*j+k+1],1);
                        }
                    }
                else
                    (*info) = k;
                }
        ipvt[n-1] = n-1;
        if (a[lda*(n-1)+(n-1)] == ZERO)
            (*info) = n-1;
        }
    }


/*
**
** DGESL benchmark
**
** We would like to declare a[][lda], but c does not allow it.  In this
** function, references to a[i][j] are written a[lda*i+j].
**
**   dgesl solves the double precision system
**   a * x = b  or  trans(a) * x = b
**   using the factors computed by dgeco or dgefa.
**
**   on entry
**
**      a       double precision[n][lda]
**              the output from dgeco or dgefa.
**
**      lda     integer
**              the leading dimension of the array  a .
**
**      n       integer
**              the order of the matrix  a .
**
**      ipvt    integer[n]
**              the pivot vector from dgeco or dgefa.
**
**      b       double precision[n]
**              the right hand side vector.
**
**      job     integer
**              = 0         to solve  a*x = b ,
**              = nonzero   to solve  trans(a)*x = b  where
**                          trans(a)  is the transpose.
**
**  on return
**
**      b       the solution vector  x .
**
**   error condition
**
**      a division by zero will occur if the input factor contains a
**      zero on the diagonal.  technically this indicates singularity
**      but it is often caused by improper arguments or improper
**      setting of lda .  it will not occur if the subroutines are
**      called correctly and if dgeco has set rcond .gt. 0.0
**      or dgefa has set info .eq. 0 .
**
**   to compute  inverse(a) * c  where  c  is a matrix
**   with  p  columns
**         dgeco(a,lda,n,ipvt,rcond,z)
**         if (!rcond is too small){
**              for (j=0,j<p,j++)
**                      dgesl(a,lda,n,ipvt,c[j][0],0);
**         }
**
**   linpack. this version dated 08/14/78 .
**   cleve moler, university of new mexico, argonne national lab.
**
**   functions
**
**   blas daxpy,ddot
*/
static void dgesl(REAL *a,int lda,int n,int *ipvt,REAL *b,int job,int roll)

    {
    REAL    t;
    int     k,kb,l,nm1;

    if (roll)
        {
        nm1 = n - 1;
        if (job == 0)
            {

            /* job = 0 , solve  a * x = b   */
            /* first solve  l*y = b         */

            if (nm1 >= 1)
                for (k = 0; k < nm1; k++)
                    {
                    l = ipvt[k];
                    t = b[l];
                    if (l != k)
                        {
                        b[l] = b[k];
                        b[k] = t;
                        }
                    daxpy_r(n-(k+1),t,&a[lda*k+k+1],1,&b[k+1],1);
                    }

            /* now solve  u*x = y */

            for (kb = 0; kb < n; kb++)
                {
                k = n - (kb + 1);
                b[k] = b[k]/a[lda*k+k];
                t = -b[k];
                daxpy_r(k,t,&a[lda*k+0],1,&b[0],1);
                }
            }
        else
            {

            /* job = nonzero, solve  trans(a) * x = b  */
            /* first solve  trans(u)*y = b             */

            for (k = 0; k < n; k++)
                {
                t = ddot_r(k,&a[lda*k+0],1,&b[0],1);
                b[k] = (b[k] - t)/a[lda*k+k];
                }

            /* now solve trans(l)*x = y     */

            if (nm1 >= 1)
                for (kb = 1; kb < nm1; kb++)
                    {
                    k = n - (kb+1);
                    b[k] = b[k] + ddot_r(n-(k+1),&a[lda*k+k+1],1,&b[k+1],1);
                    l = ipvt[k];
                    if (l != k)
                        {
                        t = b[l];
                        b[l] = b[k];
                        b[k] = t;
                        }
                    }
            }
        }
    else
        {
        nm1 = n - 1;
        if (job == 0)
            {

            /* job = 0 , solve  a * x = b   */
            /* first solve  l*y = b         */

            if (nm1 >= 1)
                for (k = 0; k < nm1; k++)
                    {
                    l = ipvt[k];
                    t = b[l];
                    if (l != k)
                        {
                        b[l] = b[k];
                        b[k] = t;
                        }
                    daxpy_ur(n-(k+1),t,&a[lda*k+k+1],1,&b[k+1],1);
                    }

            /* now solve  u*x = y */

            for (kb = 0; kb < n; kb++)
                {
                k = n - (kb + 1);
                b[k] = b[k]/a[lda*k+k];
                t = -b[k];
                daxpy_ur(k,t,&a[lda*k+0],1,&b[0],1);
                }
            }
        else
            {

            /* job = nonzero, solve  trans(a) * x = b  */
            /* first solve  trans(u)*y = b             */

            for (k = 0; k < n; k++)
                {
                t = ddot_ur(k,&a[lda*k+0],1,&b[0],1);
                b[k] = (b[k] - t)/a[lda*k+k];
                }

            /* now solve trans(l)*x = y     */

            if (nm1 >= 1)
                for (kb = 1; kb < nm1; kb++)
                    {
                    k = n - (kb+1);
                    b[k] = b[k] + ddot_ur(n-(k+1),&a[lda*k+k+1],1,&b[k+1],1);
                    l = ipvt[k];
                    if (l != k)
                        {
                        t = b[l];
                        b[l] = b[k];
                        b[k] = t;
                        }
                    }
            }
        }
    }



/*
** Constant times a vector plus a vector.
** Jack Dongarra, linpack, 3/11/78.
** ROLLED version
*/
static void daxpy_r(int n,REAL da,REAL *dx,int incx,REAL *dy,int incy)

    {
    int i,ix,iy;

    if (n <= 0)
        return;
    if (da == ZERO)
        return;

    if (incx != 1 || incy != 1)
        {

        /* code for unequal increments or equal increments != 1 */

        ix = 1;
        iy = 1;
        if(incx < 0) ix = (-n+1)*incx + 1;
        if(incy < 0)iy = (-n+1)*incy + 1;
        for (i = 0;i < n; i++)
            {
            dy[iy] = dy[iy] + da*dx[ix];
            ix = ix + incx;
            iy = iy + incy;
            }
        return;
        }

    /* code for both increments equal to 1 */

    for (i = 0;i < n; i++)
        dy[i] = dy[i] + da*dx[i];
    }


/*
** Forms the dot product of two vectors.
** Jack Dongarra, linpack, 3/11/78.
** ROLLED version
*/
static REAL ddot_r(int n,REAL *dx,int incx,REAL *dy,int incy)

    {
    REAL dtemp;
    int i,ix,iy;

    dtemp = ZERO;

    if (n <= 0)
        return(ZERO);

    if (incx != 1 || incy != 1)
        {

        /* code for unequal increments or equal increments != 1 */

        ix = 0;
        iy = 0;
        if (incx < 0) ix = (-n+1)*incx;
        if (incy < 0) iy = (-n+1)*incy;
        for (i = 0;i < n; i++)
            {
            dtemp = dtemp + dx[ix]*dy[iy];
            ix = ix + incx;
            iy = iy + incy;
            }
        return(dtemp);
        }

    /* code for both increments equal to 1 */

    for (i=0;i < n; i++)
        dtemp = dtemp + dx[i]*dy[i];
    return(dtemp);
    }


/*
** Scales a vector by a constant.
** Jack Dongarra, linpack, 3/11/78.
** ROLLED version
*/
static void dscal_r(int n,REAL da,REAL *dx,int incx)

    {
    int i,nincx;

    if (n <= 0)
        return;
    if (incx != 1)
        {

        /* code for increment not equal to 1 */

        nincx = n*incx;
        for (i = 0; i < nincx; i = i + incx)
            dx[i] = da*dx[i];
        return;
        }

    /* code for increment equal to 1 */

    for (i = 0; i < n; i++)
        dx[i] = da*dx[i];
    }


/*
** constant times a vector plus a vector.
** Jack Dongarra, linpack, 3/11/78.
** UNROLLED version
*/
static void daxpy_ur(int n,REAL da,REAL *dx,int incx,REAL *dy,int incy)

    {
    int i,ix,iy,m;

    if (n <= 0)
        return;
    if (da == ZERO)
        return;

    if (incx != 1 || incy != 1)
        {

        /* code for unequal increments or equal increments != 1 */

        ix = 1;
        iy = 1;
        if(incx < 0) ix = (-n+1)*incx + 1;
        if(incy < 0)iy = (-n+1)*incy + 1;
        for (i = 0;i < n; i++)
            {
            dy[iy] = dy[iy] + da*dx[ix];
            ix = ix + incx;
            iy = iy + incy;
            }
        return;
        }

    /* code for both increments equal to 1 */

    m = n % 4;
    if ( m != 0)
        {
        for (i = 0; i < m; i++)
            dy[i] = dy[i] + da*dx[i];
        if (n < 4)
            return;
        }
    for (i = m; i < n; i = i + 4)
        {
        dy[i] = dy[i] + da*dx[i];
        dy[i+1] = dy[i+1] + da*dx[i+1];
        dy[i+2] = dy[i+2] + da*dx[i+2];
        dy[i+3] = dy[i+3] + da*dx[i+3];
        }
    }


/*
** Forms the dot product of two vectors.
** Jack Dongarra, linpack, 3/11/78.
** UNROLLED version
*/
static REAL ddot_ur(int n,REAL *dx,int incx,REAL *dy,int incy)

    {
    REAL dtemp;
    int i,ix,iy,m;

    dtemp = ZERO;

    if (n <= 0)
        return(ZERO);

    if (incx != 1 || incy != 1)
        {

        /* code for unequal increments or equal increments != 1 */

        ix = 0;
        iy = 0;
        if (incx < 0) ix = (-n+1)*incx;
        if (incy < 0) iy = (-n+1)*incy;
        for (i = 0;i < n; i++)
            {
            dtemp = dtemp + dx[ix]*dy[iy];
            ix = ix + incx;
            iy = iy + incy;
            }
        return(dtemp);
        }

    /* code for both increments equal to 1 */

    m = n % 5;
    if (m != 0)
        {
        for (i = 0; i < m; i++)
            dtemp = dtemp + dx[i]*dy[i];
        if (n < 5)
            return(dtemp);
        }
    for (i = m; i < n; i = i + 5)
        {
        dtemp = dtemp + dx[i]*dy[i] +
        dx[i+1]*dy[i+1] + dx[i+2]*dy[i+2] +
        dx[i+3]*dy[i+3] + dx[i+4]*dy[i+4];
        }
    return(dtemp);
    }


/*
** Scales a vector by a constant.
** Jack Dongarra, linpack, 3/11/78.
** UNROLLED version
*/
static void dscal_ur(int n,REAL da,REAL *dx,int incx)

    {
    int i,m,nincx;

    if (n <= 0)
        return;
    if (incx != 1)
        {

        /* code for increment not equal to 1 */

        nincx = n*incx;
        for (i = 0; i < nincx; i = i + incx)
            dx[i] = da*dx[i];
        return;
        }

    /* code for increment equal to 1 */

    m = n % 5;
    if (m != 0)
        {
        for (i = 0; i < m; i++)
            dx[i] = da*dx[i];
        if (n < 5)
            return;
        }
    for (i = m; i < n; i = i + 5)
        {
        dx[i] = da*dx[i];
        dx[i+1] = da*dx[i+1];
        dx[i+2] = da*dx[i+2];
        dx[i+3] = da*dx[i+3];
        dx[i+4] = da*dx[i+4];
        }
    }


/*
** Finds the index of element having max. absolute value.
** Jack Dongarra, linpack, 3/11/78.
*/
static int idamax(int n,REAL *dx,int incx)

    {
    REAL dmax;
    int i, ix, itemp;

    if (n < 1)
        return(-1);
    if (n ==1 )
        return(0);
    if(incx != 1)
        {

        /* code for increment not equal to 1 */

        ix = 1;
        dmax = fabs((double)dx[0]);
        ix = ix + incx;
        for (i = 1; i < n; i++)
            {
            if(fabs((double)dx[ix]) > dmax)
                {
                itemp = i;
                dmax = fabs((double)dx[ix]);
                }
            ix = ix + incx;
            }
        }
    else
        {

        /* code for increment equal to 1 */

        itemp = 0;
        dmax = fabs((double)dx[0]);
        for (i = 1; i < n; i++)
            if(fabs((double)dx[i]) > dmax)
                {
                itemp = i;
                dmax = fabs((double)dx[i]);
                }
        }
    return (itemp);
    }


static REAL second(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC,&now)!=0) {
        fprintf(stderr,"Unable to read monotonic clock: %s\n",strerror(errno));
        exit(EXIT_FAILURE);
    }
    return (REAL)now.tv_sec+(REAL)now.tv_nsec/1.0e9;
}
