#include "stdlib.h"
#include "math.h"
#include "stdio.h"

#define PI (4.0 * atan(1.0))
#define THETAMAX 0.15

double random_double(void)
{
  return ((double) rand()) / (RAND_MAX + 1.0);
}

double random_standard_normal(void)
{
  double r1, r2;
  r1 = random_double();
  r2 = random_double();
  return sqrt(-2.0 * log(r1)) * cos(2.0 * PI * r2);
}

void random_spherical_angles(double *theta, double *phi)
{
  double x, y, z;
  x = random_standard_normal();
  y = random_standard_normal();
  z = random_standard_normal();
  *theta = atan2(sqrt(x * x + y * y), z);
  *phi = atan2(y, x);
  return;
}

void initialize_position(double source_width, double *x)
{
  x[0] = source_width * random_standard_normal();
  x[1] = source_width * random_standard_normal();
  x[2] = 0.0;
  return;
}

void initialize_momentum(double u_mag, double *u)
{
  double theta, phi;
  do
    {
      random_spherical_angles(&theta, &phi);
    } while (theta > THETAMAX);
  u[0] = u_mag * sin(theta) * cos(phi);
  u[1] = u_mag * sin(theta) * sin(phi);
  u[2] = u_mag * cos(theta);
  return;
}


void project_to_plasma(double l_source_plasma, double *x, double *u)
{
  x[0] = x[0] + l_source_plasma * u[0] / u[2];
  x[1] = x[1] + l_source_plasma * u[1] / u[2];
  return;
}

int mod(int a, int b)
{
  return (a%b + b) % b;
}

int grid_index(int n_z, int n_y, int n_x, int k, int j, int i)
{
  int i_new, j_new, k_new;
  i_new = mod(i, n_x);
  j_new = mod(j, n_y);
  k_new = mod(k, n_z);
  return k_new * n_x * n_y + j_new * n_x + i_new;
}

void boris_push(double dt, double rqm, double *b_p, double *u)
{
  // NOTE: Currently this does not include the electric field
  double tem, gamma, otsq, u_temp[3];
  tem = 0.5 * dt / rqm;
  gamma =  sqrt(1.0 + pow(u[0], 2) + pow(u[1], 2) + pow(u[2], 2));

  b_p[0] = b_p[0] * tem / gamma;
  b_p[1] = b_p[1] * tem / gamma;
  b_p[2] = b_p[2] * tem / gamma;

  u_temp[0] = u[0] + u[1] * b_p[2] - u[2] * b_p[1];
  u_temp[1] = u[1] + u[2] * b_p[0] - u[0] * b_p[2];
  u_temp[2] = u[2] + u[0] * b_p[1] - u[1] * b_p[0];

  otsq = 2.0 / (1.0 + pow(b_p[0], 2) + pow(b_p[1], 2) + pow(b_p[2], 2));

  b_p[0] = b_p[0] * otsq;
  b_p[1] = b_p[1] * otsq;
  b_p[2] = b_p[2] * otsq;

  u[0] = u[0] + u_temp[1] * b_p[2] - u_temp[2] * b_p[1];
  u[1] = u[1] + u_temp[2] * b_p[0] - u_temp[0] * b_p[2];
  u[2] = u[2] + u_temp[0] * b_p[1] - u_temp[1] * b_p[0];

  return;
}

void interpolate_fields_cic(double *b1, double *b2, double *b3,
			    int *field_grid,
			    double *x, double dx,
			    double *b_p)
{
  double delta_x, delta_y, delta_z, wx[2], wy[2], wz[2];
  int i_lower, j_lower, k_lower, grid_index_current;
  i_lower = floor(x[0] / dx);
  j_lower = floor(x[1] / dx);
  k_lower = floor(x[2] / dx);

  delta_x = x[0] / dx - (i_lower + 0.5);
  delta_y = x[1] / dx - (j_lower + 0.5);
  delta_z = x[2] / dx - (k_lower + 0.5);

  wx[0] = 0.5 - delta_x;
  wx[1] = 0.5 + delta_x;

  wy[0] = 0.5 - delta_y;
  wy[1] = 0.5 + delta_y;

  wz[0] = 0.5 - delta_z;
  wz[1] = 0.5 + delta_z;
  
  b_p[0] = 0.0;
  b_p[1] = 0.0;
  b_p[2] = 0.0;
  for (int k=0; k<2; k++) {
    for (int j=0; j<2; j++) {
      for (int i=0; i<2; i++) {
	grid_index_current =
	  grid_index(field_grid[2], field_grid[1], field_grid[0],
		     k_lower+k, j_lower+j, i_lower+i);
	b_p[0] += b1[grid_index_current] * wz[k] * wy[j] * wx[i];
       	b_p[1] += b2[grid_index_current] * wz[k] * wy[j] * wx[i];
	b_p[2] += b3[grid_index_current] * wz[k] * wy[j] * wx[i];
      }
    }
  }
  return;
}

void advance_momentum(double dt, double dx,
		      int *field_grid,
		      double *b1, double *b2, double *b3,
		      double *x, double *u,
		      double rqm)
{
  double b_p[3];
  interpolate_fields_cic(b1, b2, b3, field_grid, x, dx, b_p);
  boris_push(dt, rqm, b_p, u);
  return;
}

void advance_position(double dt, double *u, double *x)
{
  double gamma;
  gamma = sqrt(1.0 + u[0] * u[0] + u[1] * u[1] + u[2] * u[2]);
  x[0] = x[0] + dt * u[0] / gamma;
  x[1] = x[1] + dt * u[1] / gamma;
  x[2] = x[2] + dt * u[2] / gamma;
  return;
}

void propagate(double *b1, double *b2, double *b3, int *field_grid,
	       double dx, double dt, double plasma_width,
	       double *x, double *u,
	       double rqm)
{
  while (x[2] < plasma_width) {
    advance_momentum(dt, dx, field_grid, b1, b2, b3, x, u, rqm);
    advance_position(dt, u, x);
  }
  return;
}

void project_to_detector(double plasma_width, double l_plasma_detector,
			 double *x, double *u)
{
  double distance_to_detector;
  distance_to_detector = l_plasma_detector - (x[2] - plasma_width);
  x[0] = x[0] + (distance_to_detector) * u[0] / u[2];
  x[1] = x[1] + (distance_to_detector) * u[1] / u[2];
  return;
}

void deposit_particle_2d_cic(double *field, int *field_grid,
			     double charge, double x1_dx, double x2_dx)
{
  int i_ll, j_ll;
  double dx1, dx2, w_x1[2], w_x2[2];
  
  //  x1_dx = x1_dx - 0.5;
  //  x2_dx = x2_dx - 0.5;
    
  i_ll = floor(x1_dx);
  j_ll = floor(x2_dx);

  dx1 = x1_dx - (i_ll + 0.5);
  dx2 = x2_dx - (j_ll + 0.5);

  w_x1[0] = 0.5 - dx1;
  w_x1[1] = 0.5 + dx2;

  w_x2[0] = 0.5 - dx1;
  w_x2[1] = 0.5 + dx2;

  int index;
  for (int j=0; j<2; j++) {
    for (int i=0; i<2; i++) {
      index = (j_ll + j) * field_grid[0] + (i_ll + i);
      if (index >= 512*512)
	{
	  continue;
	}
      field[index] += charge * w_x1[i] * w_x2[j];
    }
  }
  return;
}


void deposit_to_detector(double *radiograph, int *radiograph_grid,
			 double radiograph_width,
			 double *x)
{
  double charge, dx, half_width, x1_dx, x2_dx;

  charge = 1.0;
  dx = radiograph_width / ((double) radiograph_grid[0]);

  half_width = radiograph_width / 2.0;
  if (fabs(x[0]) > half_width || fabs(x[1]) > half_width)
    return;
  x1_dx = (x[0] - (-1.0 * half_width)) / dx;
  x2_dx = (x[1] - (-1.0 * half_width)) / dx;

  deposit_particle_2d_cic(radiograph, radiograph_grid, charge, x1_dx, x2_dx);
  return;
}

void create_radiograph(double *b1, double *b2, double *b3, int *field_grid,
		       double dx, double dt, double *radiograph,
		       int *radiograph_grid,
		       double radiograph_width,
		       double source_width, int n_p, double u_mag,
		       double rqm,
		       double l_source_plasma, double l_plasma_detector,
		       double plasma_width,
		       int rank)
{
  srand(rank);
  double x[3], u[3];
  for (int n=0; n<n_p; n++) {
    printf("%f \n", (double) n / n_p);
    initialize_position(source_width, x);
    initialize_momentum(u_mag, u);
    project_to_plasma(l_source_plasma, x, u);
    propagate(b1, b2, b3, field_grid, dx, dt, plasma_width, x, u, rqm);
    project_to_detector(plasma_width, l_plasma_detector, x, u);
    deposit_to_detector(radiograph, radiograph_grid, radiograph_width, x);
  }
  return;
}
