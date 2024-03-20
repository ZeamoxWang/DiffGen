#pragma once
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>
#include <functional>
#include <algorithm>
#include <cassert>
#include <unordered_set>
#include <cmath>
#include "tinyxml2.h"
#include <type_traits>
#include <string>
#include "metrics.h"
#include "lodepng.h"
#include "samples.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <experimental/filesystem>


using Real = float;

#ifndef TWOD_TRI
#define TWOD_TRI

#define CLUMP_ERROR ((Real)0.1)
#define POS_REF_RATIO ((Real)0.95)
#define COL_REF_RATIO ((Real)0.6)
#define INI_ERROR ((Real)0.01)
#define PENAL_COUNT (400)
#define LINEAR_LOSS ((Real)1)

enum class MESH_TYPE
{
    INIT, GOAL
};

enum class DER_TYPE
{
    ORIGIN, HEURISTIC, HEURISTIC_LAPLACE, HEURISTIC_NORMAL, REIN, PROVED_LAPLACE, PROVED_NORMAL 
};


using namespace std;
using namespace tinyxml2;

// some basic vector operations
template <typename T>
struct Vec2 {
    T x, y;
    Vec2(T x = 0, T y = 0) : x(x), y(y) {}
    Vec2 pow(Real p) const {
        return Vec2(std::pow(x, p), std::pow(y, p));
    }
};
template <typename T>
struct Vec3 {
    T x, y, z;
    Vec3(T x = 0, T y = 0, T z = 0) : x(x), y(y), z(z) {}
    Vec3 pow(Real p) const {
        return Vec3(std::pow(x, p), std::pow(y, p), std::pow(z, p));
    }
    bool operator==(const Vec3& other) const {
        return x == other.x && y == other.y && z == other.z;
    }
    T& operator[](int index) {
        if (index == 0) return x;
        else if (index == 1) return y;
        else if (index == 2) return z;
        else throw std::out_of_range("Index out of range");
    }
    const T& operator[](int index) const {
        if (index == 0) return x;
        else if (index == 1) return y;
        else if (index == 2) return z;
        else throw std::out_of_range("Index out of range");
    }

    bool operator<(const Vec3& other) const {
        // Lexicographical Order
        if (x < other.x) return true;
        if (x > other.x) return false;
        if (y < other.y) return true;
        if (y > other.y) return false;
        return z < other.z;
    }
};
using Vec2f = Vec2<Real>;
using Vec3i = Vec3<int>;
using Vec3f = Vec3<Real>;

template <typename T>
inline T clamp(T v, T l, T u) {
    if (v < l) return l;
    else if (v > u) return u;
    return v;
}

inline void clamp(Vec3f& v, Real l, Real u) {
    v.x = (v.x > u) ? u : ((v.x < l) ? l : v.x);
    v.y = (v.y > u) ? u : ((v.y < l) ? l : v.y);
    v.z = (v.z > u) ? u : ((v.z < l) ? l : v.z);
}

#define KMEANS_ERROR (0.01)

inline Vec2f operator+(const Vec2f& v0, const Vec2f& v1) { return Vec2f{ v0.x + v1.x, v0.y + v1.y }; }
inline Vec2f& operator+=(Vec2f& v0, const Vec2f& v1) { v0.x += v1.x; v0.y += v1.y; return v0; }
inline Vec2f& operator-=(Vec2f& v0, const Vec2f& v1) { v0.x -= v1.x; v0.y -= v1.y; return v0; }
inline Vec2f operator-(const Vec2f& v0, const Vec2f& v1) { return Vec2f{ v0.x - v1.x, v0.y - v1.y }; }
inline Vec2f operator*(Real s, const Vec2f& v) { return Vec2f{ v.x * s, v.y * s }; }
inline Vec2f operator*(const Vec2f& v, Real s) { return Vec2f{ v.x * s, v.y * s }; }
inline Vec2f operator*(const Vec2f& v0, const Vec2f& v1) { return Vec2f{ v0.x * v1.x, v0.y * v1.y }; }
inline Vec2f operator/(const Vec2f& v, Real s) { return Vec2f{ v.x / s, v.y / s }; }
inline Vec2f operator/(const Vec2f& v0, const Vec2f& v1) { return Vec2f{ v0.x / v1.x, v0.y / v1.y }; }
inline Real dot(const Vec2f& v0, const Vec2f& v1) { return v0.x * v1.x + v0.y * v1.y; }
inline Real length(const Vec2f& v) { return sqrt(dot(v, v)); }
inline Vec2f normal(const Vec2f& v) { return Vec2f{ -v.y, v.x }; }
inline Vec3f operator+(const Vec3f& v0, const Vec3f& v1) { return Vec3f{ v0.x + v1.x, v0.y + v1.y, v0.z + v1.z }; }
inline Vec3f& operator+=(Vec3f& v0, const Vec3f& v1) { v0.x += v1.x; v0.y += v1.y; v0.z += v1.z; return v0; }
inline Vec3f& operator-=(Vec3f& v0, const Vec3f& v1) { v0.x -= v1.x; v0.y -= v1.y; v0.z -= v1.z; return v0; }
inline Vec3f operator-(const Vec3f& v0, const Vec3f& v1) { return Vec3f{ v0.x - v1.x, v0.y - v1.y, v0.z - v1.z }; }
inline Vec3f operator-(const Vec3f& v) { return Vec3f{ -v.x, -v.y, -v.z }; }
inline Vec3f operator*(const Vec3f& v, Real s) { return Vec3f{ v.x * s, v.y * s, v.z * s }; }
inline Vec3f operator*(Real s, const Vec3f& v) { return Vec3f{ v.x * s, v.y * s, v.z * s }; }
inline Vec3f operator*(const Vec3f& v0, const Vec3f& v1) { return Vec3f{ v0.x * v1.x, v0.y * v1.y, v0.z * v1.z }; }
inline Vec3f operator/(const Vec3f& v, Real s) { return Vec3f{ v.x / s, v.y / s, v.z / s }; }
inline Vec3f operator/(const Vec3f& v0, const Vec3f& v1) { return Vec3f{ v0.x / v1.x, v0.y / v1.y, v0.z / v1.z }; }
inline Real dot(const Vec3f& v0, const Vec3f& v1) { return v0.x * v1.x + v0.y * v1.y + v0.z * v1.z; }
inline Real length(const Vec3f& v) { return sqrt(dot(v, v)); }
inline Real l1abs(const Vec3f& v) { return abs(v.x) + abs(v.y) + abs(v.z); }


inline Real squaredDistance(const Vec3f& v1, const Vec3f& v2) {
    return dot(v1 - v2, v1 - v2);
}

// Overload addition operator for vector + vector
inline vector<Real> operator+(const vector<Real>& v0, const vector<Real>& v1) {
    vector<Real> result;
    size_t size = min(v0.size(), v1.size());

    for (size_t i = 0; i < size; ++i) {
        result.push_back(v0[i] + v1[i]);
    }

    return result;
}

// Overload scalar multiplication operator for vector * scalar
inline vector<Real> operator*(const vector<Real>& v, const Real scalar) {
    vector<Real> result;
    for (const auto& element : v) {
        result.push_back(element * scalar);
    }

    return result;
}

// Overload subtraction operator for vector - vector
inline vector<Real> operator-(const vector<Real>& v0, const vector<Real>& v1) {
    vector<Real> result;
    size_t size = min(v0.size(), v1.size());

    for (size_t i = 0; i < size; ++i) {
        result.push_back(v0[i] - v1[i]);
    }

    return result;
}

// Overload compound addition operator for vector += vector
inline vector<Real>& operator+=(vector<Real>& v0, const vector<Real>& v1) {
    size_t size = min(v0.size(), v1.size());

    for (size_t i = 0; i < size; ++i) {
        v0[i] += v1[i];
    }

    return v0;
}

// Overload compound subtraction operator for vector -= vector
inline vector<Real>& operator-=(vector<Real>& v0, const vector<Real>& v1) {
    size_t size = min(v0.size(), v1.size());

    for (size_t i = 0; i < size; ++i) {
        v0[i] -= v1[i];
    }

    return v0;
}

// Overload compound multiplication operator for vector *= scalar
inline vector<Real>& operator*=(vector<Real>& v, const Real scalar) {
    for (auto& element : v) {
        element *= scalar;
    }

    return v;
}

// Overload scalar multiplication operator for scalar * vector
inline vector<Real> operator*(const Real scalar, const vector<Real>& v) {
    vector<Real> result;
    for (const auto& element : v) {
        result.push_back(scalar * element);
    }

    return result;
}

struct Img {
    Img(int width, int height, const Vec3f& val = Vec3f{ 0, 0, 0 }) :
        width(width), height(height) {
        color.resize(width * height, val);
    }

    vector<Vec3f> color;
    int width;
    int height;

    Img(const string& filename) {
        read_png(filename);
    }

    void read_png(const std::string& filename);
    void save_png(const std::string& filename) const;
    
    // only for debug, generate gray image
    Img(vector<int> v, int width, int height): width(width), height(height){
        assert(v.size() == width * height);
        color.resize(width * height, Vec3f{ 0, 0, 0 });
        int max = *max_element(v.begin(), v.end());
        int min = *min_element(v.begin(), v.end());
        for (int i = 0; i < v.size(); i++) {
            color[i] = (Real)(v[i] - min) / (Real)(max - min) * Vec3f{ 1, 1, 1 };
        }
    }
};

// data structures for rendering
struct TriangleMesh {

    int bg_index;
    vector<Vec2f> vertices;
    vector<Vec3i> indices;
    vector<Vec3f> colors; // defined for each face
    vector<int> orders; // from top to bottom, record the index of triangles in different layers
    vector<Real> mu; // n-1 dim

    // deep copy
    TriangleMesh(const TriangleMesh& other) :bg_index(other.bg_index), vertices(other.vertices), indices(other.indices),
        colors(other.colors), orders(other.orders), mu(other.mu) {};

    TriangleMesh(const char* filename) {
        read(filename);
    }

    TriangleMesh(int n, mt19937& rng);

    TriangleMesh(MESH_TYPE m, int n, mt19937& rng);

    TriangleMesh(const TriangleMesh& mesh, Real posRefRatio, Real colRefRatio, mt19937& rng);

    TriangleMesh(int n, const Img& target, const Vec3f& bg_color, mt19937& rng);

    void print() const;
    bool check_order_validity() const {
        if (orders.size() != indices.size()) {
            cout << "Error: Order vector size does not match indices vector size.\n";
            return false;
        }

        // Check if each order index is a valid index for the indices vector
        for (int orderIndex : orders) {
            if (orderIndex < 0 || orderIndex >= indices.size()) {
                cout << "Error: Invalid order index detected: " << orderIndex << "\n";
                return false; // Invalid order index
            }
        }

        // Check if there are duplicate order indices
        unordered_set<int> uniqueOrderIndices(orders.begin(), orders.end());
        if (uniqueOrderIndices.size() != orders.size()) {
            cout << "Error: Duplicate order indices found.\n";
            return false; // Duplicate order indices found
        }

        return true;
    }

    void bgCreate(int width, int height, const Vec3f& color, int order = -1);

    bool read(const char* filename);
    bool save(const char* filename) const;
};



struct DTriangleMesh {
    DTriangleMesh(int num_vertices, int num_colors) {
        vertices.resize(num_vertices, Vec2f{ 0, 0 });
        colors.resize(num_colors, Vec3f{ 0, 0, 0 });
        mu = vector<Real>(num_colors - 1, Real{ 0 });
    }

    void clear() {
        fill(vertices.begin(), vertices.end(), Vec2f{ 0, 0 });
        fill(colors.begin(), colors.end(), Vec3f{ 0, 0, 0 });
        fill(mu.begin(), mu.end(), 0.0);
    }

    vector<Vec2f> vertices;
    vector<Vec3f> colors;
    vector<Real> mu;
};

struct Edge {
    int v0, v1; // vertex ID, v0 < v1

    Edge(int v0, int v1) : v0(min(v0, v1)), v1(max(v0, v1)) {}

    // for sorting edges
    bool operator<(const Edge& e) const {
        return this->v0 != e.v0 ? this->v0 < e.v0 : this->v1 < e.v1;
    }
};

// for sampling edges with inverse transform sampling
struct Sampler {
    vector<Real> pmf, cdf;
};

class AdamOptimizer {
private:
    vector<Vec2f> m_vertices, v_vertices;
    vector<Vec3f> m_colors, v_colors;
    vector<Real> m_mu, v_mu;
public:
    Real beta1, beta2;
    Real verticesLearningRate;
    Real colorsLearningRate;
    Real muLearningRate;
    Real epsilon;
    Real mubeta1, mubeta2;

    int t; // Iteration counter

    // Constructor
    AdamOptimizer(Real beta1 = 0.2, Real beta2 = 0.2, Real mubeta1 = 0.9, Real mubeta2 = 0.999, Real vLearningRate = 0.001, Real cLearningRate = 0.001, Real muLearningRate = 0.001, Real epsilon = 1e-8)
        : beta1(beta1), beta2(beta2), mubeta1(mubeta1), mubeta2(mubeta2), verticesLearningRate(vLearningRate), colorsLearningRate(cLearningRate), muLearningRate(muLearningRate), epsilon(epsilon), t(0) {}

    AdamOptimizer(const char* filename) {
        read(filename);
    }

    // Optimize function
    void optimize(TriangleMesh& x, const DTriangleMesh& grads);

    bool read(const char* filename);
    bool save(const char* filename) const;
};

Vec3f raytrace(const TriangleMesh& mesh,
    const Vec2f& screen_pos,
    int* hit_index);

vector<Vec3f> raytrace_with_colors(const TriangleMesh& mesh,
    const Vec2f& screen_pos,
    vector<int>& hit_layers);

void print(DER_TYPE type);

void create_video(const vector<Img>& frames, const string& output_filename, int fps);

const string str(DER_TYPE type);

cv::Mat ImgToGrayMat(const Img& input, int channel);

vector<int> partitionColors(const Img& img, int k, const Vec3f& bg_color);

class UnionFind {
public:
    UnionFind(int size) : parent(size) {
        for (int i = 0; i < size; ++i) {
            parent[i] = i;
        }
    }

    int find(int x) {
        if (parent[x] != x) {
            parent[x] = find(parent[x]);
        }
        return parent[x];
    }

    void unionSets(int x, int y) {
        int rootX = find(x);
        int rootY = find(y);

        if (rootX != rootY) {
            parent[rootX] = std::min(rootX, rootY);
            parent[rootY] = std::min(rootX, rootY);
        }
    }

private:
    std::vector<int> parent;
};

class record {
public:
    int count;
    int sum_col;
    int sum_row;
    Vec3f sum_color;
    record() :count(0), sum_col(0), sum_row(0), sum_color(Vec3f{ 0, 0, 0 }) {};
};

#endif
