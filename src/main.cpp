#include <cmath>
#include <iostream>
#include <limits>
#include <vector>
#include "solver.h"
#include <mpi.h>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int Nx_global = 256;
    const int Ny = 128;
    const int Nz = 128;
    const int ibd = 2; // one ghost layer on each side in x

    const double x0 = 0.0;
    const double x1 = 2.0 * M_PI;
    const double y0 = 0.0;
    const double y1 = 2.0 * M_PI;
    const double z0 = 0.0;
    const double z1 = 2.0 * M_PI;

    const double dx = (x1 - x0) / static_cast<double>(Nx_global);
    const double dy = (y1 - y0) / static_cast<double>(Ny);
    const double dz = (z1 - z0) / static_cast<double>(Nz);

    const double u_advection = 1.0; // velocity only in x
    const double CFL = 0.4;
    const bool use_nonuniform_x = true; // set true to enable a simple nonuniform x-grid
    const double T_end = 5.0;

    std::vector<double> x_global(Nx_global, 0.0);
    if (use_nonuniform_x) {
        // Example: piecewise spacing with a finer left half and coarser right half
        const int split = Nx_global / 2;
        const double dx_left = 0.5 * (x1 - x0) / static_cast<double>(split);
        const double dx_right = 0.3 * (x1 - x0) / static_cast<double>(Nx_global - split);
        x_global[0] = x0;
        for (int i = 1; i < Nx_global; ++i) {
            const double dx_local = (i < split) ? dx_left : dx_right;
            x_global[i] = x_global[i - 1] + dx_local;
        }
    } else {
        x_global[0] = x0;
        for (int i = 1; i < Nx_global; ++i) {
            x_global[i] = x_global[i - 1] + dx;
        }
    }

    double dx_min = std::numeric_limits<double>::max();
    for (int i = 1; i < Nx_global; ++i) {
        dx_min = std::min(dx_min, x_global[i] - x_global[i - 1]);
    }
    const double dt = CFL * dx_min / std::abs(u_advection);

    if(rank==0){
      std::cout << "dt is set to " << dt << std::endl;
    }
    
    const int base = Nx_global / size;
    const int remainder = Nx_global % size;
    const int local_nx = base + (rank < remainder ? 1 : 0);
    const int global_x_start = rank * base + std::min(rank, remainder);

    if (local_nx == 0) {
        if (rank == 0) {
            std::cerr << "Too many MPI ranks for the chosen grid." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    const std::size_t local_size = static_cast<std::size_t>(local_nx + 2 * ibd) *
                                   static_cast<std::size_t>(Ny) *
                                   static_cast<std::size_t>(Nz);
    std::vector<double> u(local_size, 0.0);

    initialize_linear_advection3d(u, local_nx, Ny, Nz, ibd, x0, dx, dy, dz,
                                  global_x_start, Nx_global, &x_global);

    if (rank == 0) {
        std::cout << "3D linear advection: Nx=" << Nx_global
                  << " Ny=" << Ny << " Nz=" << Nz
                  << " using " << size << " MPI ranks" << std::endl;
        std::cout << "dt=" << dt << " T_end=" << T_end << std::endl;
    }

    simulate_linear_advection3d(u, local_nx, Ny, Nz, ibd,
                                dx, dy, dz, x0, y0, z0,
                                global_x_start, Nx_global, u_advection, dt, T_end,
                                MPI_COMM_WORLD, &x_global);

    MPI_Finalize();
    return 0;
}
