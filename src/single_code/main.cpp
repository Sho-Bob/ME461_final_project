#include <iostream>
#include <vector>
#include "solver.h"
#include <omp.h> 
#include "vtk_writer.h"
#include <mpi.h>

int main(int argc, char** argv) {
    // const int N = 161; // 256 grid points
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    //std::cout << rank << " out of " << size << " processors." << std::endl;
    
    const int N = 257;
    const int ibd = 1;
    int ista = ibd;
    int iend = N;
    const int N_total = N + 2*ibd;
    const double x0 = 0.0, x1 = 4.0*M_PI;
    const double dx = (x1 - x0) / (N-1);
    const double CFL = 0.01;
    const double T_end = 10.0;
    const double u_advection = 1.0;
    bool inhom_grid = false;
    
    //Allocate memory for the solution and the grid
    std::vector<double> u(N+2*ibd,0.0);
    std::vector<double> x(N_total,0.0);

    // Initialize grid
    // double dx_left = 4.0 * M_PI/(N-1);
    // double dx_right = 4.0 * M_PI/(N-1);
    double dx_left = 2.0 * M_PI/128.0;
    double dx_right = 2.0 * M_PI/32.0;
    std::cout << "dx_left = " << dx_left << std::endl;
    std::cout << "dx_right = " << dx_right << std::endl;
    x[0] = x0-dx_left;
    if(inhom_grid){
        for (int i =ista; i< iend+ibd; i++){
            double dx_dummy = 0.0;
            if(i<=130){
                dx_dummy = dx_left;
            }
            else{
                dx_dummy = dx_right;
            }
            x[i] = x[i-1] + dx_dummy;
            // if(x[i]>=2.0 * M_PI){
            //     std::cout << "x[i] = " << x[i] << std::endl;
            //     std::cout << "i = " << i << std::endl;
            //     std::cout << "dx_dummy = " <<dx_dummy << std::endl;
            //     // break;
            // }
        }
    }
    else{
        for (int i =ista; i< iend+ibd; i++){
            x[i] = x[i-1] + dx;
        }
    }
    if(inhom_grid){
        x.at(iend+ibd) = x[iend] + dx_right;
    }
    else{
        x.at(iend+ibd) = x[iend] + dx_left;
    }
    std::cout << "x[0] = " << x[0] << std::endl;
    std::cout << "x[ista] = " << x[ista] << std::endl;
    std::cout << "x[iend] = " << x[iend] << std::endl;
    std::cout << "4pi = " << 4.0 * M_PI << std::endl;
    // burgers_initialize(u,N,x0,x1,dx,ibd);
    linear_advection_initialize(u,N,x,ibd);
    //Time step
    const double dt = CFL * dx_left / u_advection;
    std::cout << "dt = " << dt << std::endl;
    const int n_steps = T_end / dt;

    //Solve the equation
    // simulate_burgers1d(u,dx,CFL,T_end,ibd);
    simulate_linear_advection1d(u,x,u_advection,T_end,ibd,dt);

    return 0;
}
