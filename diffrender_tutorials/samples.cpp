#include "samples.h"

uniform_real_distribution<Real> uni_dist = uniform_real_distribution<Real>(0, 1);
normal_distribution<Real> normal_dist = normal_distribution<Real>(0.0, 1.0);

vector<vector<Real>> trans_mat = vector<vector<Real>>(0, vector<Real>(1, 0.0));

Real laplace_cdf(Real center, Real value) {
    return 0.5 * (1 + ((value > center) ? 1 : -1) * (1 - exp(-abs(center - value))));
}

Real laplace_pdf(Real center, Real value) {
    return 0.5 * exp(-abs(center - value));
}

vector<Real> clamp(const vector<Real>& a, Real error) {
    vector<Real> result;

    for (const auto& value : a) {
        if (abs(value) < error) {
            result.push_back(0.0);
        }
        else {
            result.push_back(value);
        }
    }

    return result;
}

// b = 1
Real two_centers_laplace_distribution(Real a, Real b, mt19937& rng) {
    if (a > b) {
        swap(a, b);
    }

    Real threshold = 1.0 / 2.0 / (-a + b + 1);

    Real rnd = uni_dist(rng);

    if (rnd < threshold) {
        return 0.5 * log(rnd / threshold) + a;
    }
    else if (rnd > 1 - threshold) {
        return 0.5 * log(threshold / (1.0 - rnd)) + b;
    }
    else {
        return rnd * (-a + b + 1) - 0.5 + a;
    }
}

void initialize_trans_mat(int size) {

    trans_mat = vector<vector<Real>>(size - 1, vector<Real>(size, 0.0));

    int d = size - 1;
    for (int i = 1; i < d + 1; i++) {
        for (int j = 0; j < i; j++) {
            trans_mat[d - i][j] = 1 / sqrt(i * i + i);
        }
        trans_mat[d - i][i] = -i / sqrt(i * i + i);
    }
}

// generate a random number, upper == sigma == eyes here
vector<Real> generate_normal_distribution(const vector<Real>& mu, mt19937& generator) {

    vector<Real> result(mu.size());

    for (size_t i = 0; i < mu.size(); ++i) {
        result[i] = normal_dist(generator) + mu[i];
    }

    return result;
}

// mean = 0, scale = 1
Real laplace_dist(Real uni) {
    Real a = uni - 0.5;
    Real sign = (a > 0) ? 1 : -1;
    return -sign * log(1 - 2 * abs(a));
}

vector<Real> generate_laplace_distribution(const vector<Real>& mu, mt19937& generator) {

    vector<Real> result(mu.size());

    for (size_t i = 0; i < mu.size(); ++i) {
        result[i] = laplace_dist(uni_dist(generator)) + mu[i];
    }

    return result;
}

Real normal_pdf(const vector<Real>& values, const vector<Real>& mu) {
    Real det = 1.0;

    Real z, exponent = 0.0;
    for (size_t i = 0; i < mu.size(); ++i) {
        z = values[i] - mu[i];
        exponent += z * z;
    }

    Real normalization = pow(2.0 * M_PI, -0.5 * mu.size()) / abs(det);

    Real pdf = normalization * exp(-0.5 * exponent);

    return pdf;
}

vector<Real> normal_dmudp_over_p(const vector<Real>& values, const vector<Real>& mu) {
    return values - mu;
}

vector<Real> laplace_dmudp_over_p(const vector<Real>& values, const vector<Real>& mu) {
    vector<Real> ans(values.size());
    for (int i = 0; i < values.size(); i++) {
        if (values[i] > mu[i]) {
            ans[i] = 1;
        }
        else {
            ans[i] = -1;
        }
    }
    return ans;
}

vector<Real> x_nminus1_to_x_n(const vector<Real>& x_n_1) {

    vector<Real> x_n(x_n_1.size() + 1);

    for (int i = 0; i < x_n_1.size() + 1; i++) {
        x_n[i] = 0;
        for (int j = 0; j < x_n_1.size(); j++) {
            x_n[i] += x_n_1[j] * trans_mat[j][i];
        }
    }
    return x_n;
}

vector<Real> x_n_to_x_nminus1(const vector<Real>& x_n) {

    vector<Real> x_n_1(x_n.size() - 1);

    for (int i = 0; i < x_n_1.size(); i++) {
        x_n_1[i] = 0;
        for (int j = 0; j < x_n.size(); j++) {
            x_n_1[i] += x_n[j] * trans_mat[i][j];
        }
    }
    return x_n_1;
}

vector<int> x_n_to_orders(const vector<Real>& values) {
    int k = values.size();

    vector<pair<double, int>> value_index_pairs(values.size());
    for (int i = 0; i < values.size(); ++i) {
        value_index_pairs[i] = make_pair(values[i], i);
    }

    sort(value_index_pairs.begin(), value_index_pairs.end(), [](const auto& a, const auto& b) {
        return a.first > b.first;
        });

    vector<int> result(k);
    for (int i = 0; i < k; ++i) {
        result[i] = value_index_pairs[i].second;
    }

    return result;
}


// We set sigma as 1
Real monte_carlo_p(vector<Real> center, int i, int j, int samples, mt19937& rng) {

    assert(i != j);
    assert(0 <= i && i <= center.size());
    assert(0 <= j && j <= center.size());
    if (i > j) {
        swap(i, j);
    }

    // project center to subplane x_i - x_j = 0
    Real d = abs(center[i] - center[j]) / sqrt(2);
    Real p_d = 1 / sqrt(2 * M_PI) * exp(-0.5 * d * d);
    center[i] = 0.5 * (center[i] + center[j]);

    // delete j
    if (j >= 0 && j < center.size()) {
        center.erase(center.begin() + j);
    }
    else {
        cout << "Invalid index." << endl;
    }

    // random sampling from center in n-1 dim, calculate the possibility of i is the largest one;
    Real ans = 0;
    for (int t = 0; t < samples; t++) {
        Real tmp = 1;
        Real x_i = normal_dist(rng) + center[i];

        for (int k = 0; k < center.size(); k++) {
            // center size should be shrinked
            if (k == i) continue;

            Real pivot = x_i - center[k];
            tmp *= 0.5 * (1 + std::erf(pivot / std::sqrt(2.0)));
        }
        ans += tmp / samples;
    }

    return p_d * ans;
}

// We set sigma as 1
Real monte_carlo_p_laplace(vector<Real> center, int i, int j, int samples, mt19937& rng) {

    assert(i != j);
    assert(0 <= i && i <= center.size());
    assert(0 <= j && j <= center.size());

    Real a = center[i];
    Real b = center[j];
    Real mid = 0.5 * (a + b);
    Real sigma = 1;
    if (abs(b - a) != 0) {
        sigma = abs(b - a) / sqrt(12);
    }

    // random sampling from center in n-1 dim, calculate the possibility of i is the largest one;
    Real ans = 0;
    for (int t = 0; t < samples; t++) {
        Real rnd = sigma * normal_dist(rng);
        Real x_p = mid + rnd;
        Real x_q = mid - rnd;
        Real tmp_p = 1;
        Real tmp_q = 1;

        for (int k = 0; k < center.size(); k++) {
            // center size should be shrinked
            if (k == i || k == j) {
                tmp_p *= laplace_pdf(center[k], x_p);
                tmp_q *= laplace_pdf(center[k], x_q);
            }
            else {
                tmp_p *= laplace_cdf(center[k], x_p);
                tmp_q *= laplace_cdf(center[k], x_q);
            }
        }

        Real x_pdf = 1 / sqrt(2 * M_PI) / sigma * exp(-0.5 / sigma / sigma * rnd * rnd);
        ans += (tmp_p + tmp_q) / x_pdf / 2 / samples;
    }

    return ans;
}
