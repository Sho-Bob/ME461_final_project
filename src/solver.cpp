#include "solver.h"
// #include "limiter.h"
#include "vtk_writer.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <omp.h>
#include <cassert>
#include <iostream>
#include <sys/stat.h>
#include <sys/types.h>
#ifdef _WIN32
#include <direct.h>
#endif
#include <mpi.h>

namespace {
inline int idx3d(int i, int j, int k, int Ny, int Nz) {
    return (i * Ny + j) * Nz + k;
}

inline void make_dir_if_missing(const std::string& path) {
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0755);
#endif
}

inline double global_x_at(int iglobal, int Nx_global, double x0, double dx,
                          const std::vector<double>* x_global) {
    if (!x_global || x_global->empty()) {
        return x0 + static_cast<double>(iglobal) * dx;
    }

    const int n = static_cast<int>(x_global->size());
    if (iglobal >= 0 && iglobal < n) {
        return (*x_global)[iglobal];
    }

    const double left_spacing = (n > 1) ? ((*x_global)[1] - (*x_global)[0]) : dx;
    const double right_spacing = (n > 1) ? ((*x_global)[n - 1] - (*x_global)[n - 2]) : dx;

    if (iglobal < 0) {
        return (*x_global)[0] + static_cast<double>(iglobal) * left_spacing;
    }
    // iglobal >= n
    return (*x_global)[n - 1] + static_cast<double>(iglobal - (n - 1)) * right_spacing;
}
}

void burgers_initialize(std::vector<double>& u, int N, double x0, double x1, double dx, int ibd){
    #pragma omp parallel for
    for(int i = ibd; i < N+ibd; i++){
        double x = x0 + (i-ibd) * dx;
        u[i] = std::sin(2.0 * M_PI * x);
    }
    Apply_BC(u,N,ibd);
}

void linear_advection_initialize(std::vector<double>& u,int N, std::vector<double>& x, int ibd){
    double eta1 = 4.0;
    double eta2 = 18.0;
    #pragma omp parallel for
    for(int i = 0; i < N+ibd; i++){
        u[i] = std::cos(eta2 * x[i]) * std::exp(-5.0 * std::pow(x[i]-1.5*M_PI,2));
        // Gaussian pulse
        // u[i] = std::exp(-5.0 * std::pow(x[i]-1.5*M_PI,2));
    }
    Apply_BC(u,N,ibd);
}   

void Apply_BC(std::vector<double>& u, int N, int ibd){
    /// Neumann boundary conditions
    #pragma omp parallel for
    for (int i=0; i<ibd; i++){
        u[i] = u[ibd];
        u[N+ibd+i] = u[N+ibd-1];
    }
}

inline double minmod(double a, double b) {
    double kappa = 1.0/3.0;
    double kappa_f = (3.0-kappa)/(1.0+kappa);
    if (a*b*kappa_f > 0.0)
        return std::min(std::abs(a),std::abs(b))*std::copysign(1.0,a);
    else
        return 0.0;
}

inline double central_derivative(const std::vector<double>& u,
                                 const std::vector<double>& x,
                                 int u_index,
                                 int stride = 1,
                                 int coord_index = -1) {
    const int xi = (coord_index >= 0) ? coord_index : u_index;
    assert(xi - 1 >= 0 && xi + 1 < static_cast<int>(x.size()));
    assert(u_index - stride >= 0 && u_index + stride < static_cast<int>(u.size()));
    const double dx_total = x[xi + 1] - x[xi - 1];
    assert(dx_total != 0.0);
    return (u[u_index + stride] - u[u_index - stride]) / dx_total;
}

void reconstruct_MUSCL_minmod(const std::vector<double>& u, std::vector<double>& uL, std::vector<double>& uR, int N, int ibd) {
    
        #pragma omp parallel for
        for (int i = 0; i <= N; ++i) {
            const double kappa = 1.0/3.0;
            int im2 = i + ibd - 2;
            int im1 = i + ibd - 1;
            int i0  = i + ibd;
            int ip1 = i + ibd + 1;
    
            double duL0 = minmod(u[i0] - u[im1], u[ip1] - u[i0]);
            double duL1 = minmod(u[ip1] - u[i0], u[ip1+1] - u[ip1]);
            uL[i] = u[i0] + 0.25 * ((1.0 - kappa) * duL0 + (1.0 + kappa) * duL1);
    
            double duR0 = minmod(u[ip1] - u[i0], u[ip1+1] - u[ip1]);
            double duR1 = minmod(u[ip1+1] - u[ip1], u[ip1] - u[i0]);
            uR[i] = u[ip1] - 0.25 * ((1.0 + kappa) * duR0 + (1.0 - kappa) * duR1);
        }
    
    
}

void output_terminal(std::vector<double>& u, int N, int ibd, int n_steps){
  //std::cout << "Final solsution at cell centers:" << std::endl;
		double max_u = -1e10;
		double min_u = 1e10;
	        #pragma omp parallel
		for (int i = ibd; i < N+ibd; i++){
		    #pragma omp critical
		    {
			if (u[i] > max_u) max_u = u[i];
			if (u[i] < min_u) min_u = u[i];
		    }
		}
		std::cout << "++++++++++++++++++++++" << std::endl;
		std::cout << "Step: " << n_steps << std::endl;
		std::cout << "Max u: " << max_u << std::endl;
		std::cout << "Min u: " << min_u << std::endl;
		
}

void simulate_burgers1d(std::vector<double>& u, double dx, double CFL, double T_end, int ibd){
    const int N = u.size()-2*ibd;
    int n_steps = 0;
    std::vector<double> u_new(N,0.0);
    std::vector<double> u_flux(N+1,0.0);
    std::vector<double> uL(N+1,0.0), uR(N+1,0.0);

    double t = 0.0;

    while(t < T_end){

        n_steps++;
        //Compute dt from CFL condition
        double max_speed = 0.0;
        #pragma omp parallel for reduction(max:max_speed)
        for (int i = ibd;i < N+ibd;i++){
            max_speed = std::max(max_speed,std::abs(u[i]));
        }
        double dt = CFL * dx / max_speed;

        // Reconstruct left and right states (1st order)
        #pragma omp parallel for
        for (int i=0;i < N+1;i++){
        // MUSCL for future work
            uL[i] = u[i+ibd-1];
            uR[i] = u[i+ibd];
        }

        // reconstruct_MUSCL_minmod(u, uL, uR, N, ibd);
        // reconstruct_WENO5(u, uL, uR, N, ibd);
    
        //Compute fluxes
        #pragma omp parallel for
        for (int i=0; i<N+1; i++){
            double qm = 0.5 * (uL[i] + std::abs(uL[i]));
            double qp = 0.5 * (uR[i] - std::abs(uR[i]));
            u_flux[i] = std::max(0.5 * qm*qm, 0.5 * qp*qp);
        }

        // Update solution
        #pragma omp parallel for
        for (int i=0; i<N; i++){
            u_new[i] = u[i+ibd] - dt/dx * (u_flux[i+1]-u_flux[i]);
        }

        // Update solution
        #pragma omp parallel for
        for (int i=0; i<N; i++){
            u[i+ibd] = u_new[i];
        }

        // Boundary conditions (periodic)
        Apply_BC(u,N,ibd);

        //Write solution to VTK file
        if(n_steps % 100 == 0 || n_steps == 1){
            // Create position vector for non-uniform grid support
            std::vector<double> x(u.size());
            for (int i = 0; i < u.size(); ++i) {
                x[i] = (i - ibd) * dx;
            }
            write_vtk(u,x,"burgers1d_step_" + std::to_string(n_steps) + ".vtk");
        }

        t += dt;
    }
    
}

void rk3rd_step(std::vector<double>& u,std::vector<double>& u_new,std::vector<double>& x,std::vector<double>& u_old,double dt,int stage, int ibd, double u_advection){
    int N = u.size() - 2*ibd;
    int ista = ibd;
    int iend = N;
    if(stage == 0){
        #pragma omp parallel for
        for (int i=ista;i < iend;i++){
            double dudx = central_derivative(u, x, i);
            u_new[i] = u_old[i] - dt*u_advection * dudx;
        }
        // Apply_BC(u_new,N,ibd);
    }
    else if(stage == 1){
        #pragma omp parallel for
        for (int i=ista;i < iend;i++){
            double dudx = central_derivative(u, x, i);
            u_new[i] = 0.75 * u_old[i] + 0.25 * u[i] - 0.25 * dt*u_advection * dudx;
        }
        // Apply_BC(u_new,N,ibd);
    }
    else if(stage == 2){
        #pragma omp parallel for
        for (int i=ista;i < iend;i++){
            double dudx = central_derivative(u, x, i);
            u_new[i] = 1.0/3.0 * u_old[i] + 2.0/3.0 * u[i] - 2.0/3.0 * dt*u_advection * dudx;
        }
        // Apply_BC(u_new,N,ibd);
    }
    
}

void simulate_linear_advection1d(std::vector<double>& u,std::vector<double>& x, double u_advection, double T_end, int ibd,double dt){
    int N = u.size() - 2*ibd;
    int n_steps = 0;
    int ista = ibd;
    int iend = N;
    int output_step = 1000;
    std::vector<double> u_new(N,0.0);
    std::vector<double> u_old(N,0.0);
    std::vector<double> uL(N+1,0.0), uR(N+1,0.0);

    double t = 0.0;
    // write_vtk(u,x,"linear_advection1d_step_" + std::to_string(n_steps) + ".vtk");
    write_txt(u,x,"linear_advection1d_hf_correlation_eq_" + std::to_string(n_steps) + ".txt");

    while(t<= T_end){
        u_old = u;
        for (int stage = 0;stage < 3; stage++){
            rk3rd_step(u,u_new,x,u_old,dt,stage,ibd,u_advection);
            for (int i=ista;i < iend+ibd;i++){
                u[i] = u_new[i];
            }
            Apply_BC(u,N,ibd);
        }
        t += dt;
        n_steps++;
        if(n_steps % 100 == 0 ){
            // write_vtk(u,x,"linear_advection1d_step_" + std::to_string(n_steps) + ".vtk");
            write_txt(u,x,"linear_advection1d_hf_correlation_eq_" + std::to_string(n_steps) + ".txt");
        }
	if(n_steps % output_step == 0){
	  output_terminal(u,N,ibd,n_steps);
         }
    }
}

// ----------------- 3D linear advection with MPI -----------------

static void exchange_x_halos(std::vector<double>& u, int local_nx, int Ny, int Nz,
                             int ibd, MPI_Comm comm) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const int left = (rank - 1 + size) % size;
    const int right = (rank + 1) % size;
    const std::size_t plane_size = static_cast<std::size_t>(Ny) * static_cast<std::size_t>(Nz);

    if (size == 1) {
        for (int j = 0; j < Ny; ++j) {
            for (int k = 0; k < Nz; ++k) {
                u[idx3d(0, j, k, Ny, Nz)] = u[idx3d(local_nx + ibd - 1, j, k, Ny, Nz)];
                u[idx3d(local_nx + ibd, j, k, Ny, Nz)] = u[idx3d(ibd, j, k, Ny, Nz)];
            }
        }
        return;
    }

    std::vector<double> send_left(plane_size), send_right(plane_size);
    std::vector<double> recv_left(plane_size), recv_right(plane_size);

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            const std::size_t offset = static_cast<std::size_t>(j) * Nz + k;
            send_left[offset] = u[idx3d(ibd, j, k, Ny, Nz)];
            send_right[offset] = u[idx3d(local_nx + ibd - 1, j, k, Ny, Nz)];
        }
    }

    MPI_Sendrecv(send_left.data(), plane_size, MPI_DOUBLE, left, 0,
                 recv_right.data(), plane_size, MPI_DOUBLE, right, 0,
                 comm, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_right.data(), plane_size, MPI_DOUBLE, right, 1,
                 recv_left.data(), plane_size, MPI_DOUBLE, left, 1,
                 comm, MPI_STATUS_IGNORE);

    for (int j = 0; j < Ny; ++j) {
        for (int k = 0; k < Nz; ++k) {
            const std::size_t offset = static_cast<std::size_t>(j) * Nz + k;
            u[idx3d(local_nx + ibd, j, k, Ny, Nz)] = recv_right[offset];
            u[idx3d(ibd - 1, j, k, Ny, Nz)] = recv_left[offset];
        }
    }
}

void initialize_linear_advection3d(std::vector<double>& u, std::vector<double>& v, std::vector<double>& w, int local_nx, int Ny, int Nz,
                                   int ibd, double x0, double dx, double dy, double dz,
                                   int global_x_start, int global_nx,MPI_Comm comm,
                                   const std::vector<double>* x_global) {
    const double x_min = x_global && !x_global->empty() ? x_global->front() : x0;
    const double x_max = x_global && !x_global->empty()
                         ? x_global->back()
                         : x0 + dx * static_cast<double>(global_nx - 1);
    const double Lx = x_max - x_min;
    const double Ly = dy * static_cast<double>(Ny);
    const double Lz = dz * static_cast<double>(Nz);
    const double xc = x_min + 0.5 * Lx;
    const double yc = 0.5 * Ly;
    const double zc = 0.5 * Lz;
    const bool read_data = false;

    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    
    if(read_data){
      std::ifstream data_file("./ini_data/proc_" + std::to_string(rank) + "/u_vec_ini.txt");
      for (int i_local = 0; i_local < local_nx; ++i_local) {
          const int i = i_local + ibd;
          const int iglobal = global_x_start + i_local;
          const double x = global_x_at(iglobal, global_nx, x0, dx, x_global);
          for (int j = 0; j < Ny; ++j) {
              const double y = static_cast<double>(j) * dy;
              for (int k = 0; k < Nz; ++k) {
		data_file >> u[idx3d(i, j, k, Ny, Nz)] >> v[idx3d(i, j, k, Ny, Nz)] >> w[idx3d(i, j, k, Ny, Nz)];
            }
        }
      }
      data_file.close();
    }
    else{
    for (int i_local = 0; i_local < local_nx; ++i_local) {
        const int i = i_local + ibd;
        const int iglobal = global_x_start + i_local;
        const double x = global_x_at(iglobal, global_nx, x0, dx, x_global);
        for (int j = 0; j < Ny; ++j) {
            const double y = static_cast<double>(j) * dy;
            for (int k = 0; k < Nz; ++k) {
                const double z = static_cast<double>(k) * dz;
		const double r2 = std::pow(x - xc, 2.0) + std::pow(y - yc, 2.0) + std::pow(z - zc, 2.0);
                u[idx3d(i, j, k, Ny, Nz)] = std::exp(-r2);
		v[idx3d(i, j, k, Ny, Nz)] = std::exp(-r2);
		w[idx3d(i, j, k, Ny, Nz)] = std::exp(-r2);
            }
        }
    }
    }
    exchange_x_halos(u, local_nx, Ny, Nz, ibd, MPI_COMM_WORLD);
    exchange_x_halos(v, local_nx, Ny, Nz, ibd, MPI_COMM_WORLD);
    exchange_x_halos(w, local_nx, Ny, Nz, ibd, MPI_COMM_WORLD);
}

void write_txt_3d(const std::vector<double>& u, int local_nx, int Ny, int Nz,
                  int ibd, double x0, double dx, double dy, double dz,
                  int global_x_start, int Nx_global, double time, MPI_Comm comm,
                  const std::vector<double>* x_global) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    make_dir_if_missing("./data");
    std::ostringstream dir_stream;
    dir_stream << "./data/" << rank;
    const std::string dir = dir_stream.str();
    make_dir_if_missing(dir);

    std::ostringstream filename;
    filename << dir << "/solution_t" << std::fixed << std::setprecision(6) << time << ".txt";
    std::ofstream file(filename.str().c_str());
    if (!file) {
        if (rank == 0) {
            std::cerr << "Failed to open " << filename.str() << " for writing." << std::endl;
        }
        return;
    }

    file << std::scientific << std::setprecision(8);
    file << "# time " << time << "\n";
    for (int i = ibd; i < local_nx + ibd; ++i) {
        const int iglobal = global_x_start + (i - ibd);
        const double x = global_x_at(iglobal, Nx_global, x0, dx, x_global);
        for (int j = 0; j < Ny; ++j) {
            const double y = static_cast<double>(j) * dy;
            for (int k = 0; k < Nz; ++k) {
                const double z = static_cast<double>(k) * dz;
                file << x << " " << y << " " << z << " "
                     << u[idx3d(i, j, k, Ny, Nz)] << " " << time << "\n";
            }
        }
    }
}

void write_vtk_3d(const std::vector<double>& u, int local_nx, int Ny, int Nz,
                  int ibd, double x0, double y0, double z0,
                  double dx, double dy, double dz,
                  int global_x_start, int Nx_global, double time, MPI_Comm comm,
                  const std::vector<double>* x_global) {
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    make_dir_if_missing("./data");
    std::ostringstream dir_stream;
    dir_stream << "./data/" << rank;
    const std::string dir = dir_stream.str();
    make_dir_if_missing(dir);

    std::ostringstream filename;
    filename << dir << "/solution_t" << std::fixed << std::setprecision(6) << time << ".vtk";
    std::ofstream file(filename.str().c_str());
    if (!file) {
        if (rank == 0) {
            std::cerr << "Failed to open " << filename.str() << " for writing." << std::endl;
        }
        return;
    }

    const std::size_t n_points = static_cast<std::size_t>(local_nx) *
                                 static_cast<std::size_t>(Ny) *
                                 static_cast<std::size_t>(Nz);

    file << "# vtk DataFile Version 3.0\n";
    file << "3D linear advection output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << local_nx << " " << Ny << " " << Nz << "\n";
    file << "POINTS " << n_points << " double\n";
    file << std::scientific << std::setprecision(8);

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i_local = 0; i_local < local_nx; ++i_local) {
                const int iglobal = global_x_start + i_local;
                const double x = global_x_at(iglobal, Nx_global, x0, dx, x_global);
                const double y = y0 + static_cast<double>(j) * dy;
                const double z = z0 + static_cast<double>(k) * dz;
                file << x << " " << y << " " << z << "\n";
            }
        }
    }

    file << "POINT_DATA " << n_points << "\n";
    file << "SCALARS u double 1\n";
    file << "LOOKUP_TABLE default\n";
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i_local = 0; i_local < local_nx; ++i_local) {
                const int i = ibd + i_local;
                file << u[idx3d(i, j, k, Ny, Nz)] << "\n";
            }
        }
    }
}

void write_vtk_3d_global(const std::vector<double>& u, int local_nx, int Nx_global,
                         int Ny, int Nz, int ibd,
                         double x0, double y0, double z0,
                         double dx, double dy, double dz,
                         int global_x_start, double time, MPI_Comm comm,
                         const std::vector<double>* x_global) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    std::size_t local_count = static_cast<std::size_t>(local_nx) *
                              static_cast<std::size_t>(Ny) *
                              static_cast<std::size_t>(Nz);
    std::vector<double> local_data(local_count);
    {
        std::size_t pos = 0;
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i_local = 0; i_local < local_nx; ++i_local) {
                    const int i = ibd + i_local;
                    local_data[pos++] = u[idx3d(i, j, k, Ny, Nz)];
                }
            }
        }
    }

    std::vector<int> all_local_nx;
    std::vector<int> all_x_start;
    if (rank == 0) {
        all_local_nx.resize(size);
        all_x_start.resize(size);
    }
    MPI_Gather(&local_nx, 1, MPI_INT,
               rank == 0 ? all_local_nx.data() : nullptr, 1, MPI_INT,
               0, comm);
    MPI_Gather(&global_x_start, 1, MPI_INT,
               rank == 0 ? all_x_start.data() : nullptr, 1, MPI_INT,
               0, comm);

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> gathered;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size, 0);
        int total_count = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = all_local_nx[r] * Ny * Nz;
            displs[r] = total_count;
            total_count += counts[r];
        }
        gathered.resize(static_cast<std::size_t>(total_count));
    }

    MPI_Gatherv(local_data.data(), static_cast<int>(local_count), MPI_DOUBLE,
                rank == 0 ? gathered.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);

    if (rank != 0) {
        return;
    }

    const std::size_t global_count = static_cast<std::size_t>(Nx_global) *
                                     static_cast<std::size_t>(Ny) *
                                     static_cast<std::size_t>(Nz);
    std::vector<double> global_data(global_count, 0.0);

    for (int r = 0; r < size; ++r) {
        const int nx_r = all_local_nx[r];
        const int start_r = all_x_start[r];
        const double* src = gathered.data() + displs[r];
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i_local = 0; i_local < nx_r; ++i_local) {
                    const int i_global = start_r + i_local;
                    const std::size_t global_idx =
                        (static_cast<std::size_t>(i_global) * Ny + j) * Nz + k;
                    const std::size_t local_idx =
                        (static_cast<std::size_t>(k) * Ny + j) * nx_r + i_local;
                    global_data[global_idx] = src[local_idx];
                }
            }
        }
    }

    make_dir_if_missing("./data");
    make_dir_if_missing("./data/global");
    std::ostringstream filename;
    filename << "./data/global/solution_global_t"
             << std::fixed << std::setprecision(6) << time << ".vtk";
    std::ofstream file(filename.str().c_str());
    if (!file) {
        std::cerr << "Failed to open " << filename.str() << " for writing." << std::endl;
        return;
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "Global 3D linear advection output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << Nx_global << " " << Ny << " " << Nz << "\n";
    file << "POINTS " << global_count << " double\n";
    file << std::scientific << std::setprecision(8);

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx_global; ++i) {
                const double x = global_x_at(i, Nx_global, x0, dx, x_global);
                const double y = y0 + static_cast<double>(j) * dy;
                const double z = z0 + static_cast<double>(k) * dz;
                file << x << " " << y << " " << z << "\n";
            }
        }
    }

    file << "POINT_DATA " << global_count << "\n";
    file << "SCALARS u double 1\n";
    file << "LOOKUP_TABLE default\n";

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx_global; ++i) {
                const std::size_t idx =
                    (static_cast<std::size_t>(i) * Ny + j) * Nz + k;
                file << global_data[idx] << "\n";
            }
        }
    }
}

void write_vtk_velocity_3d_global(const std::vector<double>& u,
                                  const std::vector<double>& v,
                                  const std::vector<double>& w,
                                  int local_nx, int Nx_global,
                                  int Ny, int Nz, int ibd,
                                  double x0, double y0, double z0,
                                  double dx, double dy, double dz,
                                  int global_x_start, double time, MPI_Comm comm,
                                  const std::vector<double>* x_global) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    const std::size_t local_count = static_cast<std::size_t>(local_nx) *
                                    static_cast<std::size_t>(Ny) *
                                    static_cast<std::size_t>(Nz);
    std::vector<double> local_u(local_count);
    std::vector<double> local_v(local_count);
    std::vector<double> local_w(local_count);
    {
        std::size_t pos = 0;
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i_local = 0; i_local < local_nx; ++i_local) {
                    const int i = ibd + i_local;
                    const std::size_t idx = idx3d(i, j, k, Ny, Nz);
                    local_u[pos] = u[idx];
                    local_v[pos] = v[idx];
                    local_w[pos] = w[idx];
                    ++pos;
                }
            }
        }
    }

    std::vector<int> all_local_nx;
    std::vector<int> all_x_start;
    if (rank == 0) {
        all_local_nx.resize(size);
        all_x_start.resize(size);
    }
    MPI_Gather(&local_nx, 1, MPI_INT,
               rank == 0 ? all_local_nx.data() : nullptr, 1, MPI_INT,
               0, comm);
    MPI_Gather(&global_x_start, 1, MPI_INT,
               rank == 0 ? all_x_start.data() : nullptr, 1, MPI_INT,
               0, comm);

    std::vector<int> counts;
    std::vector<int> displs;
    std::vector<double> gathered_u;
    std::vector<double> gathered_v;
    std::vector<double> gathered_w;
    if (rank == 0) {
        counts.resize(size);
        displs.resize(size, 0);
        int total_count = 0;
        for (int r = 0; r < size; ++r) {
            counts[r] = all_local_nx[r] * Ny * Nz;
            displs[r] = total_count;
            total_count += counts[r];
        }
        gathered_u.resize(static_cast<std::size_t>(total_count));
        gathered_v.resize(static_cast<std::size_t>(total_count));
        gathered_w.resize(static_cast<std::size_t>(total_count));
    }

    MPI_Gatherv(local_u.data(), static_cast<int>(local_count), MPI_DOUBLE,
                rank == 0 ? gathered_u.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_v.data(), static_cast<int>(local_count), MPI_DOUBLE,
                rank == 0 ? gathered_v.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);
    MPI_Gatherv(local_w.data(), static_cast<int>(local_count), MPI_DOUBLE,
                rank == 0 ? gathered_w.data() : nullptr,
                rank == 0 ? counts.data() : nullptr,
                rank == 0 ? displs.data() : nullptr,
                MPI_DOUBLE, 0, comm);

    if (rank != 0) {
        return;
    }

    const std::size_t global_count = static_cast<std::size_t>(Nx_global) *
                                     static_cast<std::size_t>(Ny) *
                                     static_cast<std::size_t>(Nz);
    std::vector<double> global_u(global_count, 0.0);
    std::vector<double> global_v(global_count, 0.0);
    std::vector<double> global_w(global_count, 0.0);

    for (int r = 0; r < size; ++r) {
        const int nx_r = all_local_nx[r];
        const int start_r = all_x_start[r];
        const double* src_u = gathered_u.data() + displs[r];
        const double* src_v = gathered_v.data() + displs[r];
        const double* src_w = gathered_w.data() + displs[r];
        for (int k = 0; k < Nz; ++k) {
            for (int j = 0; j < Ny; ++j) {
                for (int i_local = 0; i_local < nx_r; ++i_local) {
                    const int i_global = start_r + i_local;
                    const std::size_t global_idx =
                        (static_cast<std::size_t>(i_global) * Ny + j) * Nz + k;
                    const std::size_t local_idx =
                        (static_cast<std::size_t>(k) * Ny + j) * nx_r + i_local;
                    global_u[global_idx] = src_u[local_idx];
                    global_v[global_idx] = src_v[local_idx];
                    global_w[global_idx] = src_w[local_idx];
                }
            }
        }
    }

    make_dir_if_missing("./data");
    make_dir_if_missing("./data/global");
    std::ostringstream filename;
    filename << "./data/global/velocity_global_t"
             << std::fixed << std::setprecision(6) << time << ".vtk";
    std::ofstream file(filename.str().c_str());
    if (!file) {
        std::cerr << "Failed to open " << filename.str() << " for writing." << std::endl;
        return;
    }

    file << "# vtk DataFile Version 3.0\n";
    file << "Global 3D velocity output\n";
    file << "ASCII\n";
    file << "DATASET STRUCTURED_GRID\n";
    file << "DIMENSIONS " << Nx_global << " " << Ny << " " << Nz << "\n";
    file << "POINTS " << global_count << " double\n";
    file << std::scientific << std::setprecision(8);

    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx_global; ++i) {
                const double x = global_x_at(i, Nx_global, x0, dx, x_global);
                const double y = y0 + static_cast<double>(j) * dy;
                const double z = z0 + static_cast<double>(k) * dz;
                file << x << " " << y << " " << z << "\n";
            }
        }
    }

    file << "POINT_DATA " << global_count << "\n";
    file << "VECTORS velocity double\n";
    for (int k = 0; k < Nz; ++k) {
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx_global; ++i) {
                const std::size_t idx =
                    (static_cast<std::size_t>(i) * Ny + j) * Nz + k;
                file << global_u[idx] << " "
                     << global_v[idx] << " "
                     << global_w[idx] << "\n";
            }
        }
    }
}

void simulate_linear_advection3d(std::vector<double>& u, std::vector<double>& v, std::vector<double>& w, int local_nx, int Ny, int Nz,
                                 int ibd, double dx, double dy, double dz,
                                 double x0, double y0, double z0,
                                 int global_x_start, int Nx_global, double u_advection, double dt,
                                 double T_end, MPI_Comm comm,
                                 const std::vector<double>* x_global) {
    std::vector<double> u_new(u.size(), 0.0);
    std::vector<double> u_old(u.size(), 0.0);
    std::vector<double> v_new(v.size(), 0.0);
    std::vector<double> v_old(v.size(), 0.0);
    std::vector<double> w_new(w.size(), 0.0);
    std::vector<double> w_old(w.size(), 0.0);
    std::vector<double> x_line(static_cast<std::size_t>(local_nx + 2 * ibd), 0.0);
    for (int i = 0; i < local_nx + 2 * ibd; ++i) {
        const int iglobal = global_x_start + (i - ibd);
        x_line[i] = global_x_at(iglobal, Nx_global, x0, dx, x_global);
    }

    double t = 0.0;
    int step = 0;
    const int rank = [ & ](){int r; MPI_Comm_rank(comm, &r); return r;}();
    const int output_interval = 100;
    const int stride_x = Ny * Nz;

    // initial output at t = 0
    // write_txt_3d(u, local_nx, Ny, Nz, ibd, x0, dx, dy, dz,
    //              global_x_start, Nx_global, t, comm, x_global);
    // write_vtk_3d(u, local_nx, Ny, Nz, ibd, x0, y0, z0, dx, dy, dz,
    //              global_x_start, Nx_global, t, comm, x_global);
    write_vtk_3d_global(u, local_nx, Nx_global, Ny, Nz, ibd,
                        x0, y0, z0, dx, dy, dz, global_x_start, t, comm, x_global);
    write_vtk_velocity_3d_global(u, v, w, local_nx, Nx_global, Ny, Nz, ibd,
                                 x0, y0, z0, dx, dy, dz, global_x_start, t, comm, x_global);

    while (t < T_end - 1e-12) {
        u_old = u;
	v_old = v;
	w_old = w;
        for (int stage = 0; stage < 3; ++stage) {
            exchange_x_halos(u, local_nx, Ny, Nz, ibd, comm);
	    exchange_x_halos(v, local_nx, Ny, Nz, ibd, comm);
	    exchange_x_halos(w, local_nx, Ny, Nz, ibd, comm);
            for (int i = ibd; i < local_nx + ibd; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    for (int k = 0; k < Nz; ++k) {
                        const int idx = idx3d(i, j, k, Ny, Nz);
                        const double dudx = central_derivative(u, x_line, idx, stride_x, i);
			const double dvdx = central_derivative(v, x_line, idx, stride_x, i);
			const double dwdx = central_derivative(w, x_line, idx, stride_x, i);
                        const double u_stage = u[idx] - dt * u_advection * dudx;
			const double v_stage = v[idx] - dt * u_advection * dvdx;
			const double w_stage = w[idx] - dt * u_advection * dwdx;
                        if (stage == 0) {
                            u_new[idx] = u_stage; // u^1
			    v_new[idx] = v_stage; // v^1
			    w_new[idx] = w_stage; // w^1
                        } else if (stage == 1) {
                            u_new[idx] = 0.75 * u_old[idx] + 0.25 * u_stage; // u^2
			    v_new[idx] = 0.75 * v_old[idx] + 0.25 * v_stage; // v^2
			    w_new[idx] = 0.75 * w_old[idx] + 0.25 * w_stage; // w^2
                        } else {
                            u_new[idx] = (1.0 / 3.0) * u_old[idx] + (2.0 / 3.0) * u_stage; // u^{n+1}
			    v_new[idx] = (1.0 / 3.0) * v_old[idx] + (2.0 / 3.0) * v_stage; // v^{n+1}
			    w_new[idx] = (1.0 / 3.0) * w_old[idx] + (2.0 / 3.0) * w_stage; // w^{
                        }
                    }
                }
            }

            for (int i = ibd; i < local_nx + ibd; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    for (int k = 0; k < Nz; ++k) {
                        u[idx3d(i, j, k, Ny, Nz)] = u_new[idx3d(i, j, k, Ny, Nz)];
			v[idx3d(i, j, k, Ny, Nz)] = v_new[idx3d(i, j, k, Ny, Nz)];
			w[idx3d(i, j, k, Ny, Nz)] = w_new[idx3d(i, j, k, Ny, Nz)];
                    }
                }
            }
        }

        t += dt;
        ++step;

        if (step % output_interval == 0 || t >= T_end - 1e-12) {
            // write_txt_3d(u, local_nx, Ny, Nz, ibd, x0, dx, dy, dz,
            //              global_x_start, Nx_global, t, comm, x_global);
            // write_vtk_3d(u, local_nx, Ny, Nz, ibd, x0, y0, z0, dx, dy, dz,
            //              global_x_start, Nx_global, t, comm, x_global);
            // write_vtk_3d_global(u, local_nx, Nx_global, Ny, Nz, ibd,
            //                     x0, y0, z0, dx, dy, dz, global_x_start, t, comm, x_global);
            write_vtk_velocity_3d_global(u, v, w, local_nx, Nx_global, Ny, Nz, ibd,
                                         x0, y0, z0, dx, dy, dz, global_x_start, t, comm, x_global);
	   
        }

        if (step % 10 == 0) {
            double local_sum = 0.0;
            double local_max = -1e300;
            double local_min = 1e300;
            for (int i = ibd; i < local_nx + ibd; ++i) {
                for (int j = 0; j < Ny; ++j) {
                    for (int k = 0; k < Nz; ++k) {
                        const double val = u[idx3d(i, j, k, Ny, Nz)];
                        local_sum += val;
                        local_max = std::max(local_max, val);
                        local_min = std::min(local_min, val);
                    }
                }
            }
            double global_sum = 0.0, global_max = 0.0, global_min = 0.0;
            MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, comm);
            MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, comm);
            MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, comm);
            if (rank == 0) {
	      std::cout << "++++++++++++++++++++++" << std::endl;
	      std::cout << "step: " << step << std::endl;
	      std::cout << " t="   << t << std::endl;
	      std::cout << " min=" << global_min << std::endl;
	      std::cout << " max=" << global_max << std::endl;
            }
        }
    }
}
