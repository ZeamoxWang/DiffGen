#pragma once
#include "2d_triangles.h"
#include "samples.h"
#include <vector>

using namespace std;
using namespace tinyxml2;

using Real = float;

struct Img;
struct TriangleMesh;

Real loss_l2(const Img& a, const Img& b);

Real weighted_order_recall(const vector<int>& order, const vector<vector<Real>>& order_m,
    int samples_per_pixel,
    int height,
    int width);

Real unweighted_order_recall(const vector<int>& order, const vector<vector<Real>>& order_m,
    int samples_per_pixel,
    int height,
    int width);

vector<vector<Real>> order_matrix(const TriangleMesh& mesh,
    int samples_per_pixel,
    int height,
    int width);

Img l1diff(const Img& a, const Img& b, Real thres = 0.1);
Img l2diff(const Img& a, const Img& b, Real thres = 0.0027);