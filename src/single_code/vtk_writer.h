#ifndef VTK_WRITER_H
#define VTK_WRITER_H

#include <vector>
#include <string>

void write_vtk(const std::vector<double>& u, const std::vector<double>& x, const std::string& filename);
void write_txt(const std::vector<double>& u, const std::vector<double>& x, const std::string& filename);
#endif