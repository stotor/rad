#include <cstdlib>
#include <cmath>

#define PI (4.0 * atan(1.0))

double random_double(void)
{
  return rand() / (RAND_MAX + 1.0);
}

double random_standard_normal(void)
{
  double u1, u2;
  u1 = random_double();
  u2 = random_double();
  return sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
}

void intialize_position(double source_width,
			double &x1, double &x2, double &x3)
{
  x1 = source_width * random_standard_normal();
  x2 = source_width * random_standard_normal();
  x3 = 0;
  return;
}

void intialize_momentum(double u_mag, double &u1, double &u2, double &u3)
{
  double theta, phi;
  theta = PI * random_double();
  phi = 2.0 * PI * random_double();
  u1 = u_mag * sin(theta) * cos(phi);
  u2 = u_mag * sin(theta) * sin(phi);
  u3 = u_mag * cos(theta);
  return;
}


void project_to_plasma(double l_source_plasma,
		       double &x1, double &x2,
		       double u1, double u2, double u3)
{
  x1 = l_source_plasma * u1 / u3;
  x2 = l_source_plasma * u2 / u3;
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

void boris_push(double dt, double rqm,
		double b1_p, double b2_p, double b3_p,
		double &u1, double &u2, double &u3)
{
  // NOTE: Currently this does not include the electric field
  double tem, gamma, otsq, u1_temp, u2_temp, u3_temp;
  tem = 0.5 * dt / rqm;
  gamma =  sqrt(1.0 + u1 * u1 + u2 * u2 + u3 * u3);

  b1_p = b1_p * tem / gamma;
  b2_p = b2_p * tem / gamma;
  b3_p = b3_p * tem / gamma;

  u1_temp = u1 + u2 * b3_p - u3 * b2_p;
  u2_temp = u2 + u3 * b1_p - u1 * b3_p;
  u3_temp = u3 + u1 * b2_p - u2 * b1_p;

  otsq = 2.0 / (1.0 + b1_p * b1_p + b2_p * b2_p + b3_p * b3_p);

  b1_p = b1_p * otsq;
  b2_p = b2_p * otsq;
  b3_p = b3_p * otsq;

  u1 = u1 + u2_temp * b3_p - u3_temp * b2_p;
  u2 = u2 + u3_temp * b1_p - u1_temp * b3_p;
  u3 = u3 + u1_temp * b2_p - u2_temp * b1_p;

  return;
}

void interpolate_fields_cic(double *b1, double *b2, double *b3,
			    int *field_grid,
			    double x1, double x2, double x3, double dx,
			    double &b1_p, double &b2_p, double &b3_p)
{
  double delta_x, delta_y, delta_z, wx[2], wy[2], wz[2];
  int i_lower, j_lower, k_lower, grid_index_current;

  i_lower = floor(x1 / dx);
  j_lower = floor(x2 / dx);
  k_lower = floor(x3 / dx);

  delta_x = x1 - (i_lower + 0.5);
  delta_y = x2 - (j_lower + 0.5);
  delta_z = x3 - (k_lower + 0.5);

  wx[0] = 0.5 - delta_x;
  wx[1] = 0.5 + delta_x;

  wy[0] = 0.5 - delta_y;
  wy[1] = 0.5 + delta_y;

  wz[0] = 0.5 - delta_z;
  wz[1] = 0.5 + delta_z;
  
  b1_p = 0;
  b2_p = 0;
  b3_p = 0;
  for (int k=0; k<2; k++) {
    for (int j=0; j<2; j++) {
      for (int i=0; i<2; i++) {
	grid_index_current =
	  grid_index(field_grid[2], field_grid[1], field_grid[0],
		     k_lower+k, j_lower+j, i_lower+i);
	b1_p += b1[grid_index_current] * wz[k] * wy[j] * wx[i];
	b2_p += b2[grid_index_current] * wz[k] * wy[j] * wx[i];
	b3_p += b3[grid_index_current] * wz[k] * wy[j] * wx[i];
      }
    }
  }
  return;
}

void advance_momentum(double dt, double dx,
		      int *field_grid,
		      double *b1, double *b2, double *b3,
		      double x1, double x2, double x3,
		      double &u1, double &u2, double &u3,
		      double rqm)
{
  double b1_p, b2_p, b3_p;
  interpolate_fields_cic(b1, b2, b3,
			 field_grid,
			 x1, x2, x3, dx,
			 b1_p, b2_p, b3_p);
  boris_push(dt, rqm, b1_p, b2_p, b3_p, u1, u2, u3);
  return;
}

void advance_position(double dt,
		      double u1, double u2, double u3,
		      double &x1, double &x2, double &x3)
{
  double gamma;
  gamma = sqrt(1.0 + u1 * u1 + u2 * u2 + u3 * u3);
  x1 = x1 + dt * u1 / gamma;
  x2 = x2 + dt * u2 / gamma;
  x3 = x3 + dt * u3 / gamma;
  return;
}

void propagate(double *b1, double *b2, double *b3, int *field_grid,
	       double dx, double dt, double plasma_width,
	       double &x1, double &x2, double &x3,
	       double &u1, double &u2, double &u3,
	       double rqm)
{
  while (x3 < plasma_width) {
    advance_momentum(dt, dx, field_grid,
		     b1, b2, b3, x1, x2, x3, u1, u2, u3, rqm);
    advance_position(dt, u1, u2, u3, x1, x2, x3);
  }
  return;
}

void project_to_detector(double plasma_width, double l_plasma_detector,
			 double &x1, double &x2, double &x3,
			 double u1, double u2, double u3)
{
  double distance_to_detector;
  distance_to_detector = l_plasma_detector - (x3 - plasma_width);
  x1 = (distance_to_detector) * u1 / u3;
  x2 = (distance_to_detector) * u2 / u3;
  return;
}

void deposit_particle_2d_cic(double *field, int *field_grid,
			     double charge, double x1_dx, double x2_dx)
{
  int i_ll, j_ll;
  double dx1, dx2, w_x1[2], w_x2[2];
  
  x1_dx = x1_dx - 0.5;
  x2_dx = x2_dx - 0.5;
    
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
      index = (j_ll + j) * field_grid[0] * (i_ll + i);
      field[index] += charge * w_x1[i] * w_x2[j];
    }
  }
  return;
}


void deposit_to_detector(double *radiograph, int *radiograph_grid,
			 double radiograph_width,
			 double x1, double x2)
{
  double charge, dx, half_width, x1_dx, x2_dx;
  charge = 1.0;
  dx = radiograph_width / double(radiograph_grid[0]);
  half_width = radiograph_width / 2.0;
  if (fabs(x1) > half_width || fabs(x2) > half_width)
    return;
  x1_dx = (x1 - (-1.0 * half_width)) / dx;
  x2_dx = (x2 - (-1.0 * half_width)) / dx;
  deposit_particle_2d_cic(radiograph, radiograph_grid, charge,
			       x1_dx, x2_dx);
  return;
}

void calculate_radiograph(double *b1, double *b2, double *b3,
			  int *field_grid, double dx, double dt,
			  double *radiograph, int *radiograph_grid,
			  double radiograph_width,
			  double source_width, int n_p, double u_mag,
			  double rqm,
			  double l_source_plasma, double l_plasma_detector,
			  double plasma_width,
			  int rank)
{
  srand(rank);
  double x1, x2, x3, u1, u2, u3;
  for (int n=0; n<n_p; n++) {
    intialize_position(source_width, x1, x2, x3);
    intialize_momentum(u_mag, u1, u2, u3);
    project_to_plasma(l_source_plasma, x1, x2, u1, u2, u3);
    propagate(b1, b2, b3, field_grid, dx, dt, plasma_width,
	      x1, x2, x3, u1, u2, u3, rqm);
    project_to_detector(plasma_width, l_plasma_detector,
			x1, x2, x3, u1, u2, u3);
    deposit_to_detector(radiograph, radiograph_grid, radiograph_width, x1, x2);
  }
  return;
}
