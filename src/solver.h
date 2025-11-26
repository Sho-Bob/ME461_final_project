#ifndef SOLVER_H
#define SOLVER_H

#include <vector>
#include <mpi.h>

void burgers_initialize(std::vector<double>& u, int N, double x0, double x1, double dx, int ibd);
void linear_advection_initialize(std::vector<double>& u, int N, std::vector<double>& x, int ibd);
void simulate_burgers1d(std::vector<double>& u, double dx, double CFL, double T_end, int ibd);
void simulate_linear_advection1d(std::vector<double>& u,std::vector<double>& x, double CFL, double T_end, int ibd,double dt );
void output_terminal(std::vector<double>& u, int N, int ibd, int step);
void Apply_BC(std::vector<double>& u, int N, int ibd);
void reconstruct_MUSCL_minmod(const std::vector<double>& u, std::vector<double>& uL, std::vector<double>& uR, int N, int ibd);
void reconstruct_WENO5(const std::vector<double>& u, std::vector<double>& uL, std::vector<double>& uR, int N, int ibd);
void rk3rd_step(std::vector<double>& u,std::vector<double>& u_new,std::vector<double>& x,std::vector<double>& u_old,double dt,int stage, int ibd, double u_advection);

// 3D linear advection (MPI, advection only in x)
void initialize_linear_advection3d(std::vector<double>& u, std::vector<double>& v, std::vector<double>& w,int local_nx, int Ny, int Nz,
                                   int ibd, double x0, double dx, double dy, double dz,
                                   int global_x_start, int global_nx,MPI_Comm comm,
                                   const std::vector<double>* x_global = nullptr);
void simulate_linear_advection3d(std::vector<double>& u, std::vector<double>& v, std::vector<double>& w,  int local_nx, int Ny, int Nz,
                                 int ibd, double dx, double dy, double dz,
                                 double x0, double y0, double z0,
                                 int global_x_start, int Nx_global, double u_advection, double dt,
                                 double T_end,MPI_Comm comm,
                                 const std::vector<double>* x_global = nullptr);
void write_txt_3d(const std::vector<double>& u, int local_nx, int Ny, int Nz,
                  int ibd, double x0, double dx, double dy, double dz,
                  int global_x_start, int Nx_global, double time, MPI_Comm comm,
                  const std::vector<double>* x_global = nullptr);
void write_vtk_3d(const std::vector<double>& u, int local_nx, int Ny, int Nz,
                  int ibd, double x0, double y0, double z0,
                  double dx, double dy, double dz,
                  int global_x_start, int Nx_global, double time, MPI_Comm comm,
                  const std::vector<double>* x_global = nullptr);
void write_vtk_3d_global(const std::vector<double>& u, int local_nx, int Nx_global,
                         int Ny, int Nz, int ibd,
                         double x0, double y0, double z0,
                         double dx, double dy, double dz,
                          int global_x_start, double time, MPI_Comm comm,
                          const std::vector<double>* x_global = nullptr);
void write_vtk_velocity_3d_global(const std::vector<double>& u,
                                  const std::vector<double>& v,
                                  const std::vector<double>& w,
                                  int local_nx, int Nx_global,
                                  int Ny, int Nz, int ibd,
                                  double x0, double y0, double z0,
                                  double dx, double dy, double dz,
                                  int global_x_start, double time, MPI_Comm comm,
                                  const std::vector<double>* x_global = nullptr);

#endif
