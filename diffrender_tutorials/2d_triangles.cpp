// Compile: g++ -O3 -std=c++11 2d_triangles.cpp
#include "2d_triangles.h"

// build a discrete CDF using edge length
Sampler build_edge_sampler(const TriangleMesh& mesh,
    const vector<Edge>& edges) {
    vector<Real> pmf;
    vector<Real> cdf;
    pmf.reserve(edges.size());
    cdf.reserve(edges.size() + 1);
    cdf.push_back(0);
    for (auto edge : edges) {
        auto v0 = mesh.vertices[edge.v0];
        auto v1 = mesh.vertices[edge.v1];
        pmf.push_back(length(v1 - v0));
        cdf.push_back(pmf.back() + cdf.back());
    }
    auto length_sum = cdf.back();
    for_each(pmf.begin(), pmf.end(), [&](Real& p) {p /= length_sum; });
    for_each(cdf.begin(), cdf.end(), [&](Real& p) {p /= length_sum; });
    return Sampler{ pmf, cdf };
}

// binary search for inverting the CDF in the sampler
int sample(const Sampler& sampler, const Real u) {
    auto cdf = sampler.cdf;
    return clamp<int>(upper_bound(
        cdf.begin(), cdf.end(), u) - cdf.begin() - 1,
        0, cdf.size() - 2);
}

// given a triangle mesh, collect all edges.
vector<Edge> collect_edges(const TriangleMesh& mesh) {
    set<Edge> edges;
    for (auto index : mesh.indices) {
        edges.insert(Edge(index.x, index.y));
        edges.insert(Edge(index.y, index.z));
        edges.insert(Edge(index.z, index.x));
    }
    return vector<Edge>(edges.begin(), edges.end());
}

// trace a single ray at screen_pos, intersect with the triangle mesh.
Vec3f raytrace(const TriangleMesh& mesh,
    const Vec2f& screen_pos,
    int* hit_index = nullptr) {
    // loop over all triangles in a mesh, return the first one that hits
    for (int i: mesh.orders) {
        // check if the order index is valid
        if (i < 0 || i >= (int)mesh.indices.size()) {
            continue;
        }
        // retrieve the three vertices of a triangle
        auto index = mesh.indices[i];
        auto v0 = mesh.vertices[index.x], v1 = mesh.vertices[index.y], v2 = mesh.vertices[index.z];
        // form three half-planes: v1-v0, v2-v1, v0-v2
        // if a point is on the same side of all three half-planes, it's inside the triangle.
        auto n01 = normal(v1 - v0), n12 = normal(v2 - v1), n20 = normal(v0 - v2);
        auto side01 = dot(screen_pos - v0, n01) > 0;
        auto side12 = dot(screen_pos - v1, n12) > 0;
        auto side20 = dot(screen_pos - v2, n20) > 0;
        if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) {
            if (hit_index != nullptr) {
                *hit_index = i;
            }
            return mesh.colors[i];
        }
    }
    // return background
    if (hit_index != nullptr) {
        *hit_index = -1;
    }
    return mesh.colors[mesh.bg_index];
}

// return a color for the top layer, and modify the hit_layers
vector<Vec3f> raytrace_with_colors(const TriangleMesh& mesh,
    const Vec2f& screen_pos,
    vector<int>& hit_layers) {
    hit_layers.clear();
    vector<Vec3f> colors;
    // loop over all triangles in a mesh
    for (int i : mesh.orders) {
        // check if the order index is valid
        if (i < 0 || i >= (int)mesh.indices.size()) {
            continue;
        }
        // retrieve the three vertices of a triangle
        auto index = mesh.indices[i];
        auto v0 = mesh.vertices[index.x], v1 = mesh.vertices[index.y], v2 = mesh.vertices[index.z];
        // form three half-planes: v1-v0, v2-v1, v0-v2
        // if a point is on the same side of all three half-planes, it's inside the triangle.
        auto n01 = normal(v1 - v0), n12 = normal(v2 - v1), n20 = normal(v0 - v2);
        auto side01 = dot(screen_pos - v0, n01) > 0;
        auto side12 = dot(screen_pos - v1, n12) > 0;
        auto side20 = dot(screen_pos - v2, n20) > 0;
        if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) {
            hit_layers.push_back(i);
            colors.push_back(mesh.colors[i]);
        }
    }

    return colors;
}

void render(const TriangleMesh& mesh,
    int samples_per_pixel,
    mt19937& rng,
    Img& img) {
    assert(mesh.check_order_validity() == true);
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    for (int y = 0; y < img.height; y++) { // for each pixel
        for (int x = 0; x < img.width; x++) {
            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };
                    auto color = raytrace(mesh, screen_pos);
                    img.color[y * img.width + x] += color / samples_per_pixel;
                }
            }
        }
    }
}

void penalize_the_number(const TriangleMesh& mesh, vector<Real>& d_mu, Real(*loss)(int, int),
    vector<Real> (*dmudp_over_p)(const vector<Real>&, const vector<Real>&),
    vector<Real> (*generate_distribution)(const vector<Real>&, mt19937&),
    mt19937& rng) {
    
    vector<Real> center = x_nminus1_to_x_n(mesh.mu);
    vector<Real> sum(center.size(), 0);
    
    for (int i = 0; i < PENAL_COUNT; i++) {
        vector<Real> new_v = generate_distribution(center, rng);
        int count = 0;
        Real bg_v = new_v[mesh.bg_index];
        for (int j = 0; j < new_v.size(); j++) {
            if (new_v[j] <= bg_v)count++;
        }
        vector<Real> tmp = dmudp_over_p(new_v, center);
        // count from 1 to n
        for (int j = 0; j < sum.size(); j++) {
            sum[j] += 1.0 / (Real)PENAL_COUNT * (Real)loss(count, mesh.orders.size()) * tmp[j];
        }
    }

    d_mu += x_n_to_x_nminus1(sum);

}

int compare_colors(const vector<Vec3f>& colors, const Vec3f& target) {
    int index = -1;
    Real tmp, loss = FLT_MAX;
    for (int i = 0; i < colors.size(); i++) {
        if ((tmp = length(colors[i] - target)) < loss) {
            loss = tmp;
            index = i;
        }
    }
    return index;
}

void compute_interior_derivatives_heu_laplace(const TriangleMesh& mesh,
    int samples_per_pixel,
    const Img& adjoint,
    mt19937& rng,
    vector<Vec3f>& d_colors,
    vector<Real>& d_mu) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    fill(d_mu.begin(), d_mu.end(), 0.0);
    vector<Real> d_xn(d_mu.size() + 1, 0);
    vector<Real> xn = x_nminus1_to_x_n(mesh.mu);

    Img tmp(adjoint.width, adjoint.height);
    render(mesh, 4, rng, tmp);

    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };

                    int hit_index = -1;
                    raytrace(mesh, screen_pos, &hit_index);
                    if (hit_index != -1) {
                        // if running in parallel, use atomic add here.
                        pixel_color = tmp.color[y * adjoint.width + x];
                        // if running in parallel, use atomic add here.
                        target_color = adjoint.color[y * adjoint.width + x];
                        //adjoint.color[y * adjoint.width + x] / samples_per_pixel;
                        d_colors[hit_index] +=
                            2 * (pixel_color - target_color) / samples_per_pixel;


                        if (target_color.x != 0 || target_color.y != 0 || target_color.z != 0) {

                            // generate a color series

                            vector<int> hit_layers;
                            vector<Vec3f> colors = raytrace_with_colors(mesh, screen_pos, hit_layers);

                            if (hit_layers.size() < 2)continue;
                            assert(hit_layers.size() == colors.size());

                            int hits = hit_layers.size();
                            vector<Real> centers(hits, 0);
                            vector<Real> color_xn(hits, 0);
                            for (size_t i = 0; i < hits; i++) {
                                color_xn[i] = dot(target_color - colors[i], target_color - colors[i]);
                                centers[i] = xn[hit_layers[i]];
                            }

                            int best = compare_colors(colors, target_color);

                            for (int k = 0; k < hits; k++) {
                                for (int i = 0; i < hits; i++) {
                                    if (i == k || (i != best && k != best))continue;
                                    d_xn[hit_layers[k]] += (color_xn[k] - color_xn[i])
                                        * monte_carlo_p_laplace(centers, i, k, 2, rng) / sqrt(2) / samples_per_pixel;
                                }
                            }

                        }

                    }

                }
            }
        }
    }
    d_mu = x_n_to_x_nminus1(d_xn);

    penalize_the_number(mesh, d_mu, [](int x, int size) {
        return  -LINEAR_LOSS / (Real)size * (Real)x;
        },
        laplace_dmudp_over_p,
        generate_laplace_distribution, rng);

}

void compute_interior_derivatives_laplace(const TriangleMesh& mesh,
    int samples_per_pixel,
    const Img& adjoint,
    mt19937& rng,
    vector<Vec3f>& d_colors,
    vector<Real>& d_mu) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    fill(d_mu.begin(), d_mu.end(), 0.0);
    vector<Real> d_xn(d_mu.size() + 1, 0);
    vector<Real> xn = x_nminus1_to_x_n(mesh.mu);

    Img tmp(adjoint.width, adjoint.height);
    render(mesh, 4, rng, tmp);

    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };

                    int hit_index = -1;
                    raytrace(mesh, screen_pos, &hit_index);
                    if (hit_index != -1) {
                        // if running in parallel, use atomic add here.
                        pixel_color = tmp.color[y * adjoint.width + x];
                        // if running in parallel, use atomic add here.
                        target_color = adjoint.color[y * adjoint.width + x];
                        //adjoint.color[y * adjoint.width + x] / samples_per_pixel;
                        d_colors[hit_index] +=
                            2 * (pixel_color - target_color) / samples_per_pixel;


                        if (target_color.x != 0 || target_color.y != 0 || target_color.z != 0) {

                            // generate a color series

                            vector<int> hit_layers;
                            vector<Vec3f> colors = raytrace_with_colors(mesh, screen_pos, hit_layers);

                            if (hit_layers.size() < 2)continue;
                            assert(hit_layers.size() == colors.size());

                            int hits = hit_layers.size();
                            vector<Real> centers(hits, 0);
                            vector<Real> color_xn(hits, 0);
                            for (size_t i = 0; i < hits; i++) {
                                color_xn[i] = dot(target_color - colors[i], target_color - colors[i]);
                                centers[i] = xn[hit_layers[i]];
                            }

                            for (int k = 0; k < hits; k++) {
                                for (int i = 0; i < hits; i++) {
                                    if (i == k)continue;
                                    d_xn[hit_layers[k]] += (color_xn[k] - color_xn[i])
                                        * monte_carlo_p_laplace(centers, i, k, 2, rng) / sqrt(2) / samples_per_pixel;
                                }
                            }

                        }

                    }

                }
            }
        }
    }
    d_mu = x_n_to_x_nminus1(d_xn);

    penalize_the_number(mesh, d_mu, [](int x, int size) {
        return  -LINEAR_LOSS / (Real)size * (Real)x;
        },
        laplace_dmudp_over_p,
            generate_laplace_distribution, rng);
}

void compute_interior_derivatives_heu_proved(const TriangleMesh& mesh,
    int samples_per_pixel,
    const Img& adjoint,
    mt19937& rng,
    vector<Vec3f>& d_colors,
    vector<Real>& d_mu) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    fill(d_mu.begin(), d_mu.end(), 0.0);
    vector<Real> d_xn(d_mu.size() + 1, 0);
    vector<Real> xn = x_nminus1_to_x_n(mesh.mu);

    Img tmp(adjoint.width, adjoint.height);
    render(mesh, 4, rng, tmp);

    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };

                    int hit_index = -1;
                    raytrace(mesh, screen_pos, &hit_index);
                    if (hit_index != -1) {
                        pixel_color = tmp.color[y * adjoint.width + x];
                        // if running in parallel, use atomic add here.
                        target_color = adjoint.color[y * adjoint.width + x];
                        //adjoint.color[y * adjoint.width + x] / samples_per_pixel;
                        d_colors[hit_index] +=
                            2 * (pixel_color - target_color) / samples_per_pixel;


                        if (target_color.x != 0 || target_color.y != 0 || target_color.z != 0) {

                            // generate a color series

                            vector<int> hit_layers;
                            vector<Vec3f> colors = raytrace_with_colors(mesh, screen_pos, hit_layers);

                            if (hit_layers.size() < 2)continue;
                            assert(hit_layers.size() == colors.size());

                            int hits = hit_layers.size();
                            vector<Real> centers(hits, 0);
                            vector<Real> color_xn(hits, 0);
                            for (size_t i = 0; i < hits; i++) {
                                color_xn[i] = dot(target_color - colors[i], target_color - colors[i]);
                                centers[i] = xn[hit_layers[i]];
                            }

                            int best = compare_colors(colors, target_color);

                            for (int k = 0; k < hits; k++) {
                                for (int i = 0; i < hits; i++) {
                                    if (i == k || (i != best && k != best))continue;
                                    d_xn[hit_layers[k]] += (color_xn[k] - color_xn[i])
                                        * monte_carlo_p(centers, i, k, 1, rng) / sqrt(2) / samples_per_pixel;
                                }
                            }

                        }

                    }

                }
            }
        }
    }
    d_mu = x_n_to_x_nminus1(d_xn);
    penalize_the_number(mesh, d_mu, [](int x, int size) {
        return  -LINEAR_LOSS / (Real)size * (Real)x;
        },
        normal_dmudp_over_p,
            generate_normal_distribution, rng);
}

void compute_interior_derivatives_proved(const TriangleMesh& mesh,
    int samples_per_pixel,
    const Img& adjoint,
    mt19937& rng,
    vector<Vec3f>& d_colors,
    vector<Real>& d_mu) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    fill(d_mu.begin(), d_mu.end(), 0.0);
    vector<Real> d_xn(d_mu.size() + 1, 0);
    vector<Real> xn = x_nminus1_to_x_n(mesh.mu);

    Img tmp(adjoint.width, adjoint.height);
    render(mesh, 4, rng, tmp);

    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };

                    int hit_index = -1;
                    raytrace(mesh, screen_pos, &hit_index);
                    if (hit_index != -1) {
                        pixel_color = tmp.color[y * adjoint.width + x];
                        // if running in parallel, use atomic add here.
                        target_color = adjoint.color[y * adjoint.width + x];
                            //adjoint.color[y * adjoint.width + x] / samples_per_pixel;
                        d_colors[hit_index] +=
                            2 * (pixel_color - target_color) / samples_per_pixel;

                        
                        if (target_color.x != 0 || target_color.y != 0 || target_color.z != 0) {

                            // generate a color series
                            
                            vector<int> hit_layers;
                            vector<Vec3f> colors = raytrace_with_colors(mesh, screen_pos, hit_layers);
                            
                            if (hit_layers.size() < 2)continue;
                            assert(hit_layers.size() == colors.size());

                            int hits = hit_layers.size();
                            vector<Real> centers(hits, 0);
                            vector<Real> color_xn(hits, 0);
                            for (size_t i = 0; i < hits; i++) {
                                color_xn[i] = dot(target_color - colors[i], target_color - colors[i]); 
                                centers[i] = xn[hit_layers[i]];
                            }

                            for (int k = 0; k < hits; k++) {
                                for (int i = 0; i < hits; i++) {
                                    if (i == k)continue;
                                    d_xn[hit_layers[k]] += (color_xn[k] - color_xn[i])
                                        * monte_carlo_p(centers, i, k, 1, rng) / sqrt(2) / samples_per_pixel;
                                }
                            }

                        }

                    }

                }
            }
        }
    }
    d_mu = x_n_to_x_nminus1(d_xn);
    penalize_the_number(mesh, d_mu, [](int x, int size) {
        return  -LINEAR_LOSS / (Real)size * (Real)x;
        },
        normal_dmudp_over_p,
            generate_normal_distribution, rng);
}

void compute_edge_derivatives(
    const TriangleMesh& mesh,
    const vector<Edge>& edges,
    const Sampler& edge_sampler,
    const Img& adjoint,
    const int num_edge_samples,
    mt19937& rng,
    vector<Vec2f>& d_vertices) {
    for (int i = 0; i < num_edge_samples; i++) {
        // pick an edge
        auto edge_id = sample(edge_sampler, uni_dist(rng));
        auto edge = edges[edge_id];
        auto pmf = edge_sampler.pmf[edge_id];
        // pick a point p on the edge
        auto v0 = mesh.vertices[edge.v0];
        auto v1 = mesh.vertices[edge.v1];
        auto t = uni_dist(rng);
        auto p = v0 + t * (v1 - v0);
        auto xi = (int)p.x; auto yi = (int)p.y; // integer coordinates
        if (xi < 0 || yi < 0 || xi >= adjoint.width || yi >= adjoint.height) {
            continue;
        }
        // sample the two sides of the edge
        auto n = normal((v1 - v0) / length(v1 - v0));
        auto color_in = raytrace(mesh, p - 1e-3f * n);
        auto color_out = raytrace(mesh, p + 1e-3f * n);
        // get corresponding adjoint from the adjoint image,
        // multiply with the color difference and divide by the pdf & number of samples.
        auto pdf = pmf / (length(v1 - v0));
        auto weight = Real(1 / (pdf * Real(num_edge_samples)));

        Vec3f dI_d0x = (color_in - color_out) * (1 - t) * n.x * weight;
        Vec3f dI_d0y = (color_in - color_out) * (1 - t) * n.y * weight;
        Vec3f dI_d1x = (color_in - color_out) * t * n.x * weight;
        Vec3f dI_d1y = (color_in - color_out) * t * n.y * weight;
        Vec3f color_avg = Real(0.5) * (color_in + color_out);
        Vec3f d_color = color_avg - adjoint.color[yi * adjoint.width + xi];

        d_vertices[edge.v0] += Vec2f(dot(2 * d_color, dI_d0x), dot(2 * d_color, dI_d0y));
        d_vertices[edge.v1] += Vec2f(dot(2 * d_color, dI_d1x), dot(2 * d_color, dI_d1y));

    }
}

void compute_interior_derivatives_heuristic(const TriangleMesh& mesh,
    int samples_per_pixel,
    const Img& adjoint,
    mt19937& rng,
    vector<Vec3f>& d_colors,
    vector<Real>& d_mu) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    fill(d_mu.begin(), d_mu.end(), 0.0);
    vector<Real> d_xn(d_mu.size() + 1, 0);

    Img tmp(adjoint.width, adjoint.height);
    render(mesh, 4, rng, tmp);

    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };

                    int hit_index = -1;
                    raytrace(mesh, screen_pos, &hit_index);
                    if (hit_index != -1) {
                        // if running in parallel, use atomic add here.
                        pixel_color = tmp.color[y * adjoint.width + x];
                        // it is possible that we choose an a over b position but this pixel in target image is almost b!
                        target_color = adjoint.color[y * adjoint.width + x];
                        //adjoint.color[y * adjoint.width + x] / samples_per_pixel;
                        d_colors[hit_index] +=
                            2 * (pixel_color - target_color) / samples_per_pixel;

                        if (target_color.x != 0 || target_color.y != 0 || target_color.z != 0) {
                            // generate a sample
                            vector<int> hit_layers;
                            vector<Vec3f> colors = raytrace_with_colors(mesh, screen_pos, hit_layers);
                            colors[0] = pixel_color;
                            int best_color_index = compare_colors(colors, target_color);
                            if (hit_layers.size() >= 2 && best_color_index != 0) {// 
                                Vec3f color_avg = 0.5 * (pixel_color + colors[best_color_index]);
                                Real delta = dot(colors[best_color_index] - pixel_color, color_avg - target_color) / samples_per_pixel;
                                d_xn[hit_layers[best_color_index]] += pow(delta, 1);
                                d_xn[hit_layers[0]] -= pow(delta, 1);
                            }
                        }

                    }

                }
            }
        }
    }
    d_mu = clamp(x_n_to_x_nminus1(d_xn), CLUMP_ERROR);
}

void compute_interior_derivatives(const TriangleMesh& mesh,
    int samples_per_pixel,
    const Img& adjoint,
    mt19937& rng,
    vector<Vec3f>& d_colors,
    vector<Real>& d_mu) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    fill(d_mu.begin(), d_mu.end(), 0.0);
    vector<Real> d_xn(d_mu.size() + 1, 0);

    Img tmp(adjoint.width, adjoint.height);
    render(mesh, 4, rng, tmp);

    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };

                    int hit_index = -1;
                    raytrace(mesh, screen_pos, &hit_index);
                    if (hit_index != -1) {
                        // if running in parallel, use atomic add here.
                        pixel_color = tmp.color[y * adjoint.width + x];
                        // it is possible that we choose an a over b position but this pixel in target image is almost b!
                        target_color = adjoint.color[y * adjoint.width + x];
                        //adjoint.color[y * adjoint.width + x] / samples_per_pixel;
                        d_colors[hit_index] +=
                            2 * (pixel_color - target_color) / samples_per_pixel;

                    }

                }
            }
        }
    }
    d_mu.assign(d_mu.size(), 0);
}


// return a color for the top layer, and modify the hit_layers
Vec3f raytrace_with_order(const TriangleMesh& mesh,
    const Vec2f& screen_pos,
    const vector<int>& orders,
    vector<int>& hit_layers) {
    Vec3f top_color{ 0, 0, 0 };
    hit_layers.clear();
    // loop over all triangles in a mesh
    for (int i : orders) {
        // check if the order index is valid
        if (i < 0 || i >= (int)mesh.indices.size()) {
            continue;
        }
        // retrieve the three vertices of a triangle
        auto index = mesh.indices[i];
        auto v0 = mesh.vertices[index.x], v1 = mesh.vertices[index.y], v2 = mesh.vertices[index.z];
        // form three half-planes: v1-v0, v2-v1, v0-v2
        // if a point is on the same side of all three half-planes, it's inside the triangle.
        auto n01 = normal(v1 - v0), n12 = normal(v2 - v1), n20 = normal(v0 - v2);
        auto side01 = dot(screen_pos - v0, n01) > 0;
        auto side12 = dot(screen_pos - v1, n12) > 0;
        auto side20 = dot(screen_pos - v2, n20) > 0;
        if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) {
            hit_layers.push_back(i);
        }
    }
    if (!hit_layers.empty()) {
        top_color = mesh.colors[hit_layers[0]];
    }
    return top_color;
}

void compute_interior_derivatives_rein(const TriangleMesh& mesh,
    int samples_per_pixel,
    const Img& adjoint,
    mt19937& rng,
    vector<Vec3f>& d_colors,
    vector<Real>& d_mu) {
    auto sqrt_num_samples = (int)sqrt((Real)samples_per_pixel);
    Vec3f pixel_color, target_color;
    samples_per_pixel = sqrt_num_samples * sqrt_num_samples;
    fill(d_mu.begin(), d_mu.end(), 0.0);

    Img tmp(adjoint.width, adjoint.height);
    render(mesh, 4, rng, tmp);

    for (int y = 0; y < adjoint.height; y++) { // for each pixel
        for (int x = 0; x < adjoint.width; x++) {

            for (int dy = 0; dy < sqrt_num_samples; dy++) { // for each subpixel
                for (int dx = 0; dx < sqrt_num_samples; dx++) {
                    auto xoff = (dx + uni_dist(rng)) / sqrt_num_samples;
                    auto yoff = (dy + uni_dist(rng)) / sqrt_num_samples;
                    auto screen_pos = Vec2f{ x + xoff, y + yoff };

                    int hit_index = -1;
                    raytrace(mesh, screen_pos, &hit_index);
                    if (hit_index != -1) {
                        // if running in parallel, use atomic add here.
                        pixel_color = tmp.color[y * adjoint.width + x];
                        target_color = adjoint.color[y * adjoint.width + x];

                        d_colors[hit_index] +=
                            2 * (pixel_color - target_color) / samples_per_pixel;

                        if (target_color.x != 0 || target_color.y != 0 || target_color.z != 0) {
                            // generate a sample
                            // only few visible layers will be hit!!! Prune!!
                            vector<Real> x_nminus1 = generate_normal_distribution(mesh.mu, rng);
                            vector<int> sampled_order = x_n_to_orders(x_nminus1_to_x_n(x_nminus1));
                            vector<int> hit_layers;
                            Vec3f new_c = raytrace_with_order(mesh, screen_pos, sampled_order, hit_layers);

                            if (hit_layers.size() >= 2) {
                                new_c -= target_color;
                                d_mu += dot(new_c, new_c) / samples_per_pixel * normal_dmudp_over_p(x_nminus1, mesh.mu);
                            }
                        }

                    }

                }
            }
        }
    }
    penalize_the_number(mesh, d_mu, [](int x, int size) {
        return  -LINEAR_LOSS / (Real)size * (Real)x;
        },
        normal_dmudp_over_p,
            generate_normal_distribution, rng);
}



void d_render(DER_TYPE type,
    const TriangleMesh& mesh,
    const Img& adjoint,
    const int interior_samples_per_pixel,
    const int edge_samples_in_total,
    mt19937& rng,
    DTriangleMesh& d_mesh) {

    if (type == DER_TYPE::ORIGIN) {
        compute_interior_derivatives(mesh, interior_samples_per_pixel, adjoint,
            rng, d_mesh.colors, d_mesh.mu);
    }
    else if (type == DER_TYPE::REIN) {
        compute_interior_derivatives_rein(mesh, interior_samples_per_pixel, adjoint,
            rng, d_mesh.colors, d_mesh.mu);
    }
    else if (type == DER_TYPE::HEURISTIC) {
        compute_interior_derivatives_heuristic(mesh, interior_samples_per_pixel, adjoint,
            rng, d_mesh.colors, d_mesh.mu);
    }
    else if (type == DER_TYPE::PROVED_LAPLACE) {
        compute_interior_derivatives_laplace(mesh, interior_samples_per_pixel, adjoint,
            rng, d_mesh.colors, d_mesh.mu);
    }
    else if (type == DER_TYPE::PROVED_NORMAL) {
        compute_interior_derivatives_proved(mesh, interior_samples_per_pixel, adjoint,
            rng, d_mesh.colors, d_mesh.mu);
    }
    else if (type == DER_TYPE::HEURISTIC_NORMAL) {
        compute_interior_derivatives_heu_proved(mesh, interior_samples_per_pixel, adjoint,
            rng, d_mesh.colors, d_mesh.mu);
    }
    else if (type == DER_TYPE::HEURISTIC_LAPLACE) {
        compute_interior_derivatives_heu_laplace(mesh, interior_samples_per_pixel, adjoint,
            rng, d_mesh.colors, d_mesh.mu);
    }

    auto edges = collect_edges(mesh);
    auto edge_sampler = build_edge_sampler(mesh, edges);
    compute_edge_derivatives(mesh, edges, edge_sampler, adjoint, edge_samples_in_total,
        rng, d_mesh.vertices);
}

void AdamOptimizer::optimize(TriangleMesh& x, const DTriangleMesh& grads) {

    vector<Vec2f>& vertices = x.vertices;
    vector<Vec3f>& colors = x.colors;
    vector<Real>& mu = x.mu;
    const vector<Vec2f>& d_vertices = grads.vertices;
    const vector<Vec3f>& d_colors = grads.colors;
    const vector<Real>& d_mu = grads.mu;

    assert(vertices.size() == d_vertices.size());
    assert(colors.size() == d_colors.size());
    assert(mu.size() == d_mu.size());

    if (m_vertices.empty()) {
        m_vertices.resize(x.vertices.size(), Vec2f{ 0.0, 0.0 });
        v_vertices.resize(x.vertices.size(), Vec2f{ 0.0, 0.0 });
    }
    if (m_colors.empty()) {
        m_colors.resize(x.colors.size(), Vec3f{ 0.0, 0.0, 0.0 });
        v_colors.resize(x.colors.size(), Vec3f{ 0.0, 0.0, 0.0 });
    }
    if (m_mu.empty()) {
        m_mu.resize(x.mu.size(), 0.0);
        v_mu.resize(x.mu.size(), 0.0);
    }

    t++;
    Vec2f e2(epsilon, epsilon);
    Vec3f e3(epsilon, epsilon, epsilon);
    // Update m and v parameters
    for (size_t i = 0; i < vertices.size(); ++i) {
        // skip bg
        if (i == x.bg_index)continue;
        // Update m for vertices
        m_vertices[i] = beta1 * m_vertices[i] + (1 - beta1) * d_vertices[i];

        // Update v for vertices
        v_vertices[i] = beta2 * v_vertices[i] + (1 - beta2) * d_vertices[i].pow((Real)2);

        // Bias correction for m and v for vertices
        Vec2f m_hat_vertices = m_vertices[i] / (1 - pow(beta1, t));
        Vec2f v_hat_vertices = v_vertices[i] / (1 - pow(beta2, t));

        // Update vertices

        vertices[i] -= verticesLearningRate * m_hat_vertices / (v_hat_vertices.pow((Real)0.5) + e2);
        
    }

    for (size_t i = 0; i < colors.size(); ++i) {
        // skip bg
        //if (i == x.bg_index)continue;
        // Update m for colors
        m_colors[i] = beta1 * m_colors[i] + (1 - beta1) * d_colors[i];

        // Update v for colors
        v_colors[i] = beta2 * v_colors[i] + (1 - beta2) * d_colors[i].pow((Real)2);

        // Bias correction for m and v for colors
        Vec3f m_hat_colors = m_colors[i] / (1 - pow(beta1, t));
        Vec3f v_hat_colors = v_colors[i] / (1 - pow(beta2, t));

        // Update colors
        colors[i] -= colorsLearningRate * m_hat_colors / (v_hat_colors.pow((Real)0.5) + e3);
        clamp(colors[i], 0.0, 1.0);
    }

    for (size_t i = 0; i < mu.size(); ++i) {
        // Update m for mu
        m_mu[i] = mubeta1 * m_mu[i] + (1 - mubeta1) * d_mu[i];

        // Update v for mu
        v_mu[i] = mubeta2 * v_mu[i] + (1 - mubeta2) * d_mu[i] * d_mu[i];

        // Bias correction for m and v for mu
        Real m_hat_mu = m_mu[i] / (1 - pow(mubeta1, t));
        Real v_hat_mu = v_mu[i] / (1 - pow(mubeta2, t));

        // Update mu
        mu[i] -= muLearningRate * m_hat_mu / (sqrt(v_hat_mu) + epsilon);
    }

    x.orders = x_n_to_orders(x_nminus1_to_x_n(mu));
}


TriangleMesh optimization(DER_TYPE type, TriangleMesh mesh, const vector<vector<Real>>& order_gt, const string& filename,
    AdamOptimizer optimizer, mt19937& rng, int iter) {

    /* This function uses L2 to calculate the gradient*/
    Img target(filename);
    DTriangleMesh d_mesh(mesh.vertices.size(), mesh.colors.size());

    vector<Img> frames;
    Img tmp(target.width, target.height);
    render(mesh, 4, rng, tmp);
    frames.push_back(tmp);

    for (size_t i = 0; i < iter; i++) {
        cout << "size:" << mesh.orders.size() << endl;
        cout << "epoch:" << i << endl;

        d_mesh.clear();
        d_render(type, mesh, target, 4 /* interior_samples_per_pixel */,
            target.width * target.height /* edge_samples_in_total */, rng, d_mesh);
        optimizer.optimize(mesh, d_mesh);

        Img tmp(target.width, target.height);
        render(mesh, 4, rng, tmp);
        frames.push_back(tmp);

        print(type);
        cout << "loss:" << loss_l2(tmp, target) << endl;
        print(type);
        cout << "weighted order recall:" <<
            weighted_order_recall(mesh.orders, order_gt,
                4, target.height, target.width) << endl;
        print(type);
        cout << "unweighted order recall:" << unweighted_order_recall(mesh.orders, order_gt,
            4, target.height, target.width) << endl;
        print(type);
        cout << "dmu:";
        for (size_t i = 0; i < d_mesh.mu.size(); i++) {
            cout << d_mesh.mu[i] << " ";
        }
        cout << endl;

        //mesh.print();

        cout << "\n\n" << endl;
    }

    string name = "num" + to_string(mesh.mu.size()) + "iter" + to_string(iter) + str(type) + ".mp4";
    create_video(frames, name, 10);

    return mesh;
}

int regenerate(TriangleMesh& mesh, const Img& now, const Img& target, int max_regen_layers, int bin = 10) {

    if (mesh.orders[mesh.orders.size() - 1] == mesh.bg_index) return 0;

    Img graydiff = l1diff(now, target);
    //Img graydiff = l2diff(now, target);
    vector<int> assignments = partitionColors(graydiff, bin, mesh.colors[mesh.bg_index]);


    // evaluate largest connected area
    // the cv library only supports binary image, so I have to implement mine
    // two pass
    vector<vector<int>> labels(target.height, vector<int>(target.width, 0));
    int label = 0;
    vector<pair<int, int>> equal;
    for (int row = 0; row < target.height; row++) {
        for (int col = 0; col < target.width; col++) {
            int uplabel = 0;
            int leftlabel = 0;
            if (row > 0)uplabel = labels[row - 1][col];
            if (col > 0)leftlabel = labels[row][col - 1];

            int pos = row * target.width + col;
            if (assignments[pos] == 0) {
                labels[row][col] == 0;
                continue;
            }
            // now assignments[pos] != 0
            int upassign = 0;
            int leftassign = 0;
            int myassign = assignments[pos];
            if (row > 0)upassign = assignments[(row - 1) * target.width + col];
            if (col > 0)leftassign = assignments[row * target.width + col - 1];
            if (myassign != upassign && myassign != leftassign) {
                label++;
                labels[row][col] = label;
            }
            else if (myassign == leftassign && myassign != upassign) {
                labels[row][col] = leftlabel;
            }
            else if (myassign != leftassign && myassign == upassign) {
                labels[row][col] = uplabel;
            }
            else {
                labels[row][col] = min(uplabel, leftlabel);
                if (uplabel != leftlabel)equal.push_back({ uplabel, leftlabel });
            }
        }
    }

    UnionFind uf(label + 1);
    for (pair<int, int> item : equal) {
        uf.unionSets(item.first, item.second);
    }

    vector<record> rec(label + 1, record());
    for (int row = 0; row < target.height; row++) {
        for (int col = 0; col < target.width; col++) {
            int this_label = uf.find(labels[row][col]);
            rec[this_label].count++;
            rec[this_label].sum_col += col;
            rec[this_label].sum_row += row;
            rec[this_label].sum_color += target.color[row * target.width + col];
        }
    }


    // dessert rec[0]
    rec.erase(rec.begin());

    // sort acoording to area
    sort(rec.begin(), rec.end(), [](const record& a, const record& b) {
        return b.count < a.count;
        });


    // set triangles
    vector<Real> x_n = x_nminus1_to_x_n(mesh.mu);
    int changed = 0;
    for (int i = mesh.orders.size() - 1; i > mesh.orders.size() - max_regen_layers - 1; i--){
        int tri_index = mesh.orders[i];
        if (tri_index == mesh.bg_index) break;
        if (rec[changed].count == 0)break;

        // calculate the average color
        mesh.colors[tri_index] = rec[changed].sum_color / (Real)rec[changed].count;
        // lift it up to the background
        //x_n[tri_index] = x_n[mesh.bg_index];
        // lift it up to the middle
        x_n[tri_index] = 0;
        // lift it up to the top
        //x_n[tri_index] = *max_element(x_n.begin(), x_n.end());

        // take the vertices out
        Vec2f& p1 = mesh.vertices[mesh.indices[tri_index].x];
        Vec2f& p2 = mesh.vertices[mesh.indices[tri_index].y];
        Vec2f& p3 = mesh.vertices[mesh.indices[tri_index].z];

        Real r = sqrt(rec[changed].count * 4.0 / 3.0 / sqrt(3));
        Real center_x = rec[changed].sum_col / rec[changed].count;
        Real center_y = rec[changed].sum_row / rec[changed].count;

        p1 = Vec2f{ (Real)(center_x + r * sqrt(3) / 2),(Real)(center_y - 0.5 * r) };
        p2 = Vec2f{ (Real)(center_x - r * sqrt(3) / 2),(Real)(center_y - 0.5 * r) };
        p3 = Vec2f{ center_x, (Real)(center_y + r) };

        cout << "\nColors:\n";
        cout << "(" << mesh.colors[tri_index].x << ", " << mesh.colors[tri_index].y << ", " << mesh.colors[tri_index].z << ")\n";

        cout << "Vertices:\n";
        cout << "(" << p1.x << ", " << p1.y << ")   ";

        cout << "(" << p2.x << ", " << p2.y << ")   ";
        cout << "(" << p3.x << ", " << p3.y << ")  \n";
        changed++;
    }
    mesh.mu = x_n_to_x_nminus1(x_n);
    mesh.orders = x_n_to_orders(x_n);
    return changed;
}

TriangleMesh optimization(DER_TYPE type, TriangleMesh mesh, const string& filename,
    AdamOptimizer optimizer, mt19937& rng, int iter, int freq = 10, int max_regen_layers = 1) {

    /* This function uses L2 to calculate the gradient*/
    Img target(filename);
    DTriangleMesh d_mesh(mesh.vertices.size(), mesh.colors.size());

    vector<Img> frames;
    Img tmp(target.width, target.height);
    render(mesh, 4, rng, tmp);
    frames.push_back(tmp);

    for (size_t i = 0; i < iter; i++) {
        cout << "size:" << mesh.orders.size() << endl;
        cout << "epoch:" << i << endl;

        d_mesh.clear();
        d_render(type, mesh, target, 4 /* interior_samples_per_pixel */,
            target.width * target.height /* edge_samples_in_total */, rng, d_mesh);
        optimizer.optimize(mesh, d_mesh);

        Img tmp(target.width, target.height);
        render(mesh, 4, rng, tmp);

        // regenerate layers hidden by background
        // from bottom to background, modify no more than max_regen_layers
        if (i % freq == freq - 1) {
            regenerate(mesh, tmp, target, max_regen_layers, 100);
            tmp.color.assign(tmp.width * tmp.height, 0);
            render(mesh, 4, rng, tmp);
        }
        frames.push_back(tmp);

        print(type);
        cout << "loss:" << loss_l2(tmp, target) << endl;

        print(type);
        int pos = mesh.orders.size() - 1;
        for (; pos >=0 ; pos--) {
            if (mesh.orders[pos] == mesh.bg_index)break;
        }
        cout << "bg order:" << pos << endl;

        //mesh.print();

        cout << "\n\n" << endl;
    }

    string name = "num" + to_string(mesh.mu.size()) + "iter" + to_string(iter) + str(type) + ".mp4";
    create_video(frames, name, 10);

    return mesh;
}

int main(int argc, char* argv[]) {

    auto now = std::chrono::system_clock::now();

    auto timestamp = std::chrono::system_clock::to_time_t(now);
    std::tm tm;
    localtime_s(&tm, &timestamp);
    std::stringstream folderName;
    folderName << std::put_time(&tm, "%Y%m%d_%H%M%S");

    experimental::filesystem::create_directory(folderName.str());
    experimental::filesystem::current_path(folderName.str());

    ofstream file("output.txt");

    if (!file.is_open()) {
        std::cerr << "Cannot Open Doc!" << std::endl;
        return 1;
    }

    streambuf* original_cout = std::cout.rdbuf();
    cout.rdbuf(file.rdbuf());

    mt19937 rng(42);
    for (int n = 20; n <= 100; n+=20) {


        string filename = "../target.png";
        Img target(filename);

        TriangleMesh mesh0(n, target, Vec3f{ 1, 1, 1 }, rng);
        Img img_ini(target.width, target.height);
        render(mesh0, 4 /* samples_per_pixel */, rng, img_ini);
        img_ini.save_png("input" + to_string(n) + ".png");

        initialize_trans_mat(mesh0.indices.size());

        mesh0.print();

        for (int i = static_cast<int>(DER_TYPE::ORIGIN); i <= static_cast<int>(DER_TYPE::PROVED_NORMAL); ++i) {
            DER_TYPE type = static_cast<DER_TYPE>(i);
            TriangleMesh mesh = mesh0;
            int epoch = 500;


            if (type == DER_TYPE::ORIGIN) {
                cout << "It's origin.." << endl;
            }
            else if (type == DER_TYPE::REIN) {
                cout << "It's rein.." << endl;
            }
            else if (type == DER_TYPE::HEURISTIC) {
                cout << "It's heuristic.." << endl;
            }
            else if (type == DER_TYPE::PROVED_LAPLACE) {
                cout << "It's proved laplacian..." << endl;
            }
            else if (type == DER_TYPE::PROVED_NORMAL) {
                cout << "It's proved normal..." << endl;
            }
            else if (type == DER_TYPE::HEURISTIC_LAPLACE) {
               cout << "It's heuristic laplace..." << endl;
            }
            else if (type == DER_TYPE::HEURISTIC_NORMAL) {
                cout << "It's heuristic normal..." << endl;
            }

            Real beta1 = 0.8;
            Real beta2 = 0.9;
            Real verticesLearningRate = 0.4;
            Real colorsLearningRate = 0.010;

            Real mubeta1 = 0.8;
            Real mubeta2 = 0.9;
            Real muLearningRate = 1;
            Real epsilon = 1e-8;
            AdamOptimizer optimizer(beta1, beta2, mubeta1, mubeta2, verticesLearningRate, colorsLearningRate, muLearningRate, epsilon);

            /* start optimization */

            TriangleMesh outcome = optimization(type, mesh, filename, optimizer, rng, epoch, 51, 3);

            Img img_out(target.width, target.height);
            render(outcome, 4 /* samples_per_pixel */, rng, img_out);
            img_out.save_png("outcome" + to_string(n) + "_" + str(type) + ".png");
        }
    }

    cout.rdbuf(original_cout);

    file.close();
    
    return 0;
}