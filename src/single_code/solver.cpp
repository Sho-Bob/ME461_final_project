#include "solver.h"
// #include "limiter.h"
#include "vtk_writer.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <omp.h>
#include <cassert>
#include <iostream>

void burgers_initialize(std::vector<double>& u, int N, double x0, double x1, double dx, int ibd){
    #pragma omp parallel for
    for(int i = ibd; i < N+ibd; i++){
        double x = x0 + (i-ibd) * dx;
        u[i] = std::sin(2.0 * M_PI * x);
    }
    Apply_BC(u,N,ibd);
}

void linear_advection_initialize(std::vector<double>& u, int N, std::vector<double>& x, int ibd){
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
  //std::cout << "Final solution at cell centers:" << std::endl;
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
            double dxi = x[i+1] - x[i];
            double dxi_1 = x[i] - x[i-1];
            double dxi2 = std::pow(x[i+1] - x[i],2);
            double dxi_12 = std::pow(x[i] - x[i-1],2);
            // double dudx = (dxi_12*u[i+1] -dxi2*u[i-1] + (dxi2-dxi_12)*u[i])/(dxi*dxi_1*(dxi+dxi_1));
            double dudx = (u[i+1] - u[i-1])/(x[i+1] - x[i-1]);
            u_new[i] = u_old[i] - dt*u_advection * dudx;
        }
        // Apply_BC(u_new,N,ibd);
    }
    else if(stage == 1){
        #pragma omp parallel for
        for (int i=ista;i < iend;i++){
            double dxi = x[i+1] - x[i];
            double dxi_1 = x[i] - x[i-1];
            double dxi2 = std::pow(x[i+1] - x[i],2);
            double dxi_12 = std::pow(x[i] - x[i-1],2);
            // double dudx = (dxi_12*u[i+1] -dxi2*u[i-1] + (dxi2-dxi_12)*u[i])/(dxi*dxi_1*(dxi+dxi_1));
            double dudx = (u[i+1] - u[i-1])/(x[i+1] - x[i-1]);
            u_new[i] = 0.75 * u_old[i] + 0.25 * u[i] - 0.25 * dt*u_advection * dudx;
        }
        // Apply_BC(u_new,N,ibd);
    }
    else if(stage == 2){
        #pragma omp parallel for
        for (int i=ista;i < iend;i++){
            double dxi = x[i+1] - x[i];
            double dxi_1 = x[i] - x[i-1];
            double dxi2 = std::pow(x[i+1] - x[i],2);
            double dxi_12 = std::pow(x[i] - x[i-1],2);
            // double dudx = (dxi_12*u[i+1] -dxi2*u[i-1] + (dxi2-dxi_12)*u[i])/(dxi*dxi_1*(dxi+dxi_1));
            double dudx = (u[i+1] - u[i-1])/(x[i+1] - x[i-1]);
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
