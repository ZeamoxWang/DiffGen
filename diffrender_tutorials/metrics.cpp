#include "metrics.h"

Real loss_l2(const Img& a, const Img& b) {
    if (a.width != b.width || a.height != b.height) {
        cerr << "Error: Images must have the same dimensions for L2 loss calculation." << endl;
        return -1.0f;
    }

    Real loss = 0;

    for (int i = 0; i < a.color.size(); ++i) {
        Real diffX = a.color[i].x - b.color[i].x;
        Real diffY = a.color[i].y - b.color[i].y;
        Real diffZ = a.color[i].z - b.color[i].z;

        loss += diffX * diffX + diffY * diffY + diffZ * diffZ;
    }

    return sqrt(loss / a.color.size());
}

// The output is actually a gray image
Img l1diff(const Img& a, const Img& b, Real thres) {
    assert(a.width == b.width && a.height == b.height);

    Img ans(a.width, a.height);

    for (int i = 0; i < ans.color.size(); ++i) {
        Real value = l1abs(a.color[i] - b.color[i]);
        if (value > thres) {
            ans.color[i].x = ans.color[i].y = ans.color[i].z = value;
        }
        else {
            ans.color[i].x = ans.color[i].y = ans.color[i].z = 0;
        }

    }

    return ans;
}

Img l2diff(const Img& a, const Img& b, Real thres) {
    assert(a.width == b.width && a.height == b.height);

    Img ans(a.width, a.height);

    for (int i = 0; i < ans.color.size(); ++i) {
        Real value = squaredDistance(a.color[i], b.color[i]);
        if (value > thres) {
            ans.color[i].x = ans.color[i].y = ans.color[i].z = value;
        }
        else {
            ans.color[i].x = ans.color[i].y = ans.color[i].z = 0;
        }

    }

    return ans;
}

// This function generates the ground truth of orders:
// The first index is the top layer, the second index is the covered layer
// The real number records the frequency of the row layer is over column layer
// i.e. if this order pair cannot be satisfied what proportion of the area would be affected.
vector<vector<Real>> order_matrix(const TriangleMesh& mesh,
    int samples_per_pixel,
    int height,
    int width) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;

    int triangle_numbers = mesh.orders.size();

    vector<vector<Real>> ans(triangle_numbers, vector<Real>(triangle_numbers, Real(0)));

    int total_n = height * width * samples_per_pixel;

    for (int y = 0; y < height; y++) { // for each pixel
        for (int x = 0; x < width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    Real xoff = (dx + (Real)0.5) / sqrt_num_samples;
                    Real yoff = (dy + (Real)0.5) / sqrt_num_samples;
                    Vec2f screen_pos = Vec2f{ x + xoff, y + yoff };

                    vector<int> hit_layers;
                    raytrace_with_colors(mesh, screen_pos, hit_layers);

                    if(hit_layers.size()>=2){
                        int top = hit_layers[0];
                        for (int i = 1; i < hit_layers.size(); i++) {
                            ans[top][hit_layers[i]] += 1.0 / (Real)total_n;
                        }

                    }

                }
            }
        }
    }

    return ans;
}

Real unweighted_order_recall(const vector<int>& order, const vector<vector<Real>>& order_m,
    int samples_per_pixel,
    int height,
    int width) {

    assert(order.size() == order_m.size());


    vector<int> reference_order(order.size(), 0);

    for (int i = 0; i < order.size(); i++) {
        reference_order[order[i]] = i;
    }

    int count = 0;
    int satisfied = 0;

    for (int i = 0; i < order_m.size(); i++) {
        for (int j = 0; j < order_m[i].size(); j++) {
            if (order_m[i][j] > 0) {
                count++;
                if (reference_order[i] < reference_order[j]) {
                    satisfied++;
                }
            }
        }
    }

    return (Real)satisfied / (Real)count;
}

// Because of the position might not be very correct, this stuck might be caused by positions
Real weighted_order_recall(const vector<int>& order, const vector<vector<Real>>& order_m,
    int samples_per_pixel,
    int height,
    int width) {
    assert(order.size() == order_m.size());

    vector<int> reference_order(order.size(), 0);

    for (int i = 0; i < order.size(); i++) {
        reference_order[order[i]] = i;
    }

    Real count = 0;
    Real satisfied = 0;

    for (int i = 0; i < order_m.size(); i++) {
        for (int j = 0; j < order_m[i].size(); j++) {
            if (order_m[i][j] > 0) {
                count += order_m[i][j];
                if (reference_order[i] < reference_order[j]) {
                    satisfied += order_m[i][j];
                }
            }
        }
    }

    return satisfied / count;
}