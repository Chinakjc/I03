#include "dim.cuh"
#include "cuda_check.cuh"
#include "user.cuh"

#include "timer.hxx"
#include "iteration.hxx"



// A completer : definition du noyau

__global__ void test_kernel(){

    int nx = d_n[0];
    int ny = d_n[1];
    int nz = d_n[2];
    double dx = d_dx[0];
    double dy = d_dx[1];
    double dz = d_dx[2];

    double xmin = d_xmin[0];
    double ymin = d_xmin[1];
    double zmin = d_xmin[2];

    printf("nx = %d ny = %d nz = %d\ndx = %f dy = %f dz = %f\nxmin = %f ymin = %f zmin = %f\n",nx,ny,nz,dx,dy,dz,xmin,ymin,zmin);

    double lam_x = d_lambda[0];
    double lam_y = d_lambda[1];
    double lam_z = d_lambda[2];

    printf("lam_x = %f lam_x = %f lam_z = %f",lam_x,lam_y,lam_z);
    
}
__global__ void iteration_kernel(
    double*u, double *v, const double dt,
    const int imin, const int imax, 
    const int jmin, const int jmax,
    const int kmin, const int kmax
    ) {

    int nx = d_n[0];
    int ny = d_n[1];
    //int nz = d_n[2];

    double dx = d_dx[0];
    double dy = d_dx[1];
    double dz = d_dx[2];

    double xmin = d_xmin[0];
    double ymin = d_xmin[1];
    double zmin = d_xmin[2];

    int i = blockIdx.x * blockDim.x + threadIdx.x + imin;
    int j = blockIdx.y * blockDim.y + threadIdx.y + jmin;
    int k = blockIdx.z * blockDim.z + threadIdx.z + kmin;

    int n1 = nx;
    int n2 = ny * nx;

    if (i > imax || j > jmax || k > kmax)
        return;


    double lam_x = d_lambda[0];
    double lam_y = d_lambda[1];
    double lam_z = d_lambda[2];

    


    double du1 = (-2 * u[i + n1*j + n2*k] + u[i+1 + n1*j + n2*k] + u[i-1 + n1*j + n2*k]) * lam_x
            + (-2 * u[i + n1*j + n2*k] + u[i + n1*(j+1) + n2*k] + u[i + n1*(j-1) + n2*k]) * lam_y
            + (-2 * u[i + n1*j + n2*k] + u[i + n1*j + n2*(k+1)] + u[i + n1*j + n2*(k-1)]) * lam_z;

    double x = xmin + i * dx;
    double y = ymin + j * dy;
    double z = zmin + k * dz;
    double du2 = force(x, y, z);

    double du = dt * (du1 + du2);
    v[i + n1*j + n2*k] = u[i + n1*j + n2*k] + du;

}

__global__ void iteration_kernel_coarse(
    double*u, double *v, const double dt,
    const int imin, const int imax, 
    const int jmin, const int jmax,
    const int kmin, const int kmax,
    const int sx, const int sy, const int sz
    ) {

    int nx = d_n[0];
    int ny = d_n[1];
    //int nz = d_n[2];

    double dx = d_dx[0];
    double dy = d_dx[1];
    double dz = d_dx[2];

    double xmin = d_xmin[0];
    double ymin = d_xmin[1];
    double zmin = d_xmin[2];

    int i = sx *(blockIdx.x * blockDim.x + threadIdx.x) + imin;
    int j = sy *(blockIdx.y * blockDim.y + threadIdx.y) + jmin;
    int k = sz *(blockIdx.z * blockDim.z + threadIdx.z) + kmin;

    int n1 = nx;
    int n2 = ny * nx;

    if (i > imax || j > jmax || k > kmax)
        return;


    double lam_x = d_lambda[0];
    double lam_y = d_lambda[1];
    double lam_z = d_lambda[2];
    
    int ii_max = min(i + sx, imax + 1);
    int jj_max = min(j + sy, jmax + 1);
    int kk_max = min(k + sz, kmax + 1);

    
    for(int kk = k; kk < kk_max; kk++){
        for(int jj = j; jj < jj_max; jj++){
            for(int ii = i; ii < ii_max; ii++){
                double du1 = (-2 * u[ii + n1*jj + n2*kk] + u[ii+1 + n1*jj + n2*kk] + u[ii-1 + n1*jj + n2*kk]) * lam_x
            + (-2 * u[ii + n1*jj + n2*kk] + u[ii + n1*(jj+1) + n2*kk] + u[ii + n1*(jj-1) + n2*kk]) * lam_y
            + (-2 * u[ii + n1*jj + n2*kk] + u[ii + n1*jj + n2*(kk+1)] + u[ii + n1*jj + n2*(kk-1)]) * lam_z;

            double x = xmin + ii * dx;
            double y = ymin + jj * dy;
            double z = zmin + kk * dz;
            double du2 = force(x, y, z);

            double du = dt * (du1 + du2);
            v[ii + n1*jj + n2*kk] = u[ii + n1*jj + n2*kk] + du;
            }
        }
    }

    

}


int get_xyz_size_opt(const unsigned int blockSize, const unsigned int ijk_size, const unsigned int N){
    return ceil((float)ijk_size/(float)(blockSize*N));
}

dim3 get_gridSize(const dim3& blockSize, const dim3& ijk_size, const dim3& xyz_size = dim3(1,1,1)){
    int bx = blockSize.x;
    int by = blockSize.y;
    int bz = blockSize.z;


    int si = ijk_size.x;
    int sj = ijk_size.y;
    int sk = ijk_size.z;
    

    int sx = max(1,xyz_size.x);
    int sy = max(1,xyz_size.y);
    int sz = max(1,xyz_size.z);


    dim3 gridSize(ceil((float)(si) / (float)(bx * sx)),
              ceil((float)(sj) / (float)(by * sy)),
              ceil((float)(sk) / (float)(bz * sz)));

    //printf("s_ijk / (blocSize * s_xyz)\n%.3f %.3f %.3f\n",(float)(si) / (float)(bx * sx),(float)(sj) / (float)(by * sy),(float)(sk) / (float)(bz * sz));
    //printf("gridSize\n%d %d %d\n",gridSize.x,gridSize.y,gridSize.z);

    return gridSize;
}



void test_dim_cu(){
    test_kernel<<<1,1>>>();
}

void test_xyz_size_opt(const unsigned int ni, const unsigned int blockSize,bool simple=false){
    int blockSize_max = ceil((float)(ni) / (float)(blockSize));
    int rec = 0;
    for(int n=1; n<= blockSize_max; n++){
        int sx = get_xyz_size_opt(blockSize,ni,n);
        if(sx==rec&&simple){
            continue;
        }else{
            rec = sx;
        }
        float gsf = (float)(ni) / (float)(blockSize * sx);
        int gs = ceil(gsf);
        float eff = 100.0 * gsf /(float)gs;
        printf("n = %d, sx = %d, gridSize = %d efficiency = %.1f %%\n",n,sx,gs,eff);
    }
}

void test_block_size(const unsigned int ni = 399, unsigned int blockSize_max = 256){
    for(int th = 1; th <= blockSize_max; th++){
        printf("_______________________________\n");
        printf("blockSize = %d\n",th);
        test_xyz_size_opt(ni,th,true);
    }
}
void iteration(
    Values & v, Values & u, double dt, int n[3],
    int imin, int imax, 
    int jmin, int jmax,
    int kmin, int kmax)
{
// A completer : appel du noyau

    dim3 ijk_size(imax - imin + 1,jmax - jmin + 1,kmax - kmin + 1);

    const bool is_coarse = false;

    double* d_u = u.dataGPU();
    double* d_v = v.dataGPU();

    if(is_coarse){
        unsigned int bx = 57;
        unsigned int by = 7;
        unsigned int bz = 1;
        dim3 blockSize(bx,by,bz); // block size can be adjusted

        dim3 xyz_size_opt(
        get_xyz_size_opt(bx,ijk_size.x,1),
        get_xyz_size_opt(by,ijk_size.y,19),
        get_xyz_size_opt(bz,ijk_size.z,400));

        dim3 gridSize = get_gridSize(blockSize,ijk_size,xyz_size_opt);
        int sx = xyz_size_opt.x;
        int sy = xyz_size_opt.y;
        int sz = xyz_size_opt.z;
        iteration_kernel_coarse<<<gridSize, blockSize>>>(
        d_u, d_v, dt, imin, imax, jmin, jmax, kmin, kmax, sx,sy,sz);
    }else{
        unsigned int bx = 8;
        unsigned int by = 8;
        unsigned int bz = 8;
        dim3 blockSize(bx,by,bz); // block size can be adjusted
        dim3 gridSize = get_gridSize(blockSize,ijk_size);
        iteration_kernel<<<gridSize, blockSize>>>(
        d_u, d_v, dt, imin, imax, jmin, jmax, kmin, kmax);
    }
    cudaDeviceSynchronize(); // synchronize to ensure kernel execution finishes

}
