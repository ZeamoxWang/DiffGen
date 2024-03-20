#pragma once

#include <random>
#include <set>
#include <vector>
#include "2d_triangles.h"

using namespace std;
using Real = float;

Real laplace_cdf(Real center, Real value);

Real laplace_pdf(Real center, Real value);

vector<Real> clamp(const vector<Real>& a, Real error);


extern uniform_real_distribution<Real> uni_dist;
extern normal_distribution<Real> normal_dist;

// b = 1
Real two_centers_laplace_distribution(Real a, Real b, mt19937& rng);
extern vector<vector<Real>> trans_mat;
void initialize_trans_mat(int size);

// generate a random number, upper == sigma == eyes here
vector<Real> generate_normal_distribution(const vector<Real>& mu, mt19937& generator);

vector<Real> generate_laplace_distribution(const vector<Real>& mu, mt19937& generator);

Real normal_pdf(const vector<Real>& values, const vector<Real>& mu);

vector<Real> normal_dmudp_over_p(const vector<Real>& values, const vector<Real>& mu);

vector<Real> laplace_dmudp_over_p(const vector<Real>& values, const vector<Real>& mu);

vector<Real> x_nminus1_to_x_n(const vector<Real>& x_n_1);

vector<Real> x_n_to_x_nminus1(const vector<Real>& x_n);

vector<int> x_n_to_orders(const vector<Real>& values);

Real monte_carlo_p(vector<Real> center, int i, int j, int samples, mt19937& rng);

Real monte_carlo_p_laplace(vector<Real> center, int i, int j, int samples, mt19937& rng);