#include "2d_triangles.h"

template <typename T>
T clamp(T v, T l, T u);

uniform_real_distribution<Real> distribution(0.0, 1.0);
normal_distribution<Real> nor_dis(0.0, 1.0);


void print(DER_TYPE type) {
    cout << str(type);
}

const string str(DER_TYPE type) {
    switch (type) {
    case DER_TYPE::ORIGIN:
        return "ORIGIN";
        break;
    case DER_TYPE::REIN:
        return "REIN";
        break;
    case DER_TYPE::HEURISTIC:
        return "HEURISTIC";
        break;
    case DER_TYPE::PROVED_LAPLACE:
        return "PROVED_LAPLACE";
        break;
    case DER_TYPE::PROVED_NORMAL:
        return "PROVED_NORMAL";
        break;
    case DER_TYPE::HEURISTIC_NORMAL:
        return "HEURISTIC_NORMAL";
        break;
    case DER_TYPE::HEURISTIC_LAPLACE:
        return "HEURISTIC_LAPLACE";
        break;
    }
}

void TriangleMesh::print() const {

    cout << "Background Color: " << "(" << colors[bg_index].x << ", " << colors[bg_index].y << ", " << colors[bg_index].z << ")\n";

    cout << "Background Index: " << bg_index << "\n\n";

    cout << "Vertices:\n";
    int i = 0;
    for (const auto& vertex : vertices) {
        cout << "(" << vertex.x << ", " << vertex.y << ")   ";
        i++;
        if (i % 3 == 0) {
            cout << endl;
        }
    }

    cout << "\nIndices:\n";

    for (const auto& index : indices) {
        cout << "(" << index.x << ", " << index.y << ", " << index.z << ")\n";
    }

    cout << "\nColors:\n";
    for (const auto& color : colors) {
        cout << "(" << color.x << ", " << color.y << ", " << color.z << ")\n";
    }

    cout << "\nMu:\n";
    for (const auto& value : mu) {
        cout << value << " ";
    }
    cout << endl;

    cout << "\nN-d Space:\n";
    vector<Real> coordinate = x_nminus1_to_x_n(mu);
    for (const auto& value : coordinate) {
        cout << value << " ";
    }
    cout << endl;

    cout << "\nOrder:\n";
    for (const auto& value : orders) {
        cout << value << " ";
    }

    cout << "\n" << endl;

}

// PNG uses BGR!!!
void Img::save_png(const std::string& filename) const {
    std::vector<unsigned char> image;
    image.reserve(width * height * 4);

    auto tonemap = [](Real x) {
        return int(pow(clamp(x, Real(0), Real(1)), Real(1 / 2.2)) * 255 + Real(.5)); };

    for (const auto& pixel : color) {
        image.push_back(static_cast<unsigned char>(tonemap(pixel.z)));
        image.push_back(static_cast<unsigned char>(tonemap(pixel.y)));
        image.push_back(static_cast<unsigned char>(tonemap(pixel.x)));
        image.push_back(255);  // Alpha channel
    }

    unsigned error = lodepng::encode(filename.c_str(), image, width, height);
    if (error) {
        std::cerr << "Error writing PNG: " << lodepng_error_text(error) << std::endl;
    }
}

// PNG uses BGR!!!
void Img::read_png(const std::string& filename) {
    std::vector<unsigned char> image;
    unsigned png_width, png_height;

    unsigned error = lodepng::decode(image, png_width, png_height, filename.c_str());
    if (error) {
        std::cerr << "Error reading PNG: " << lodepng_error_text(error) << std::endl;
        return;
    }

    width = static_cast<int>(png_width);
    height = static_cast<int>(png_height);
    color.resize(width * height);

    auto inverseTonemap = [](Real y) {
        Real x = pow((y / 255.0), Real(2.2));
        x = clamp(x, Real(0), Real(1));
        return x;
    };

    for (size_t i = 0; i < color.size(); ++i) {
        color[i].z = static_cast<float>(image[i * 4]);
        color[i].y = static_cast<float>(image[i * 4 + 1]);
        color[i].x = static_cast<float>(image[i * 4 + 2]);

        // Apply gamma correction during reading
        color[i].z = inverseTonemap(color[i].z);
        color[i].y = inverseTonemap(color[i].y);
        color[i].x = inverseTonemap(color[i].x);
    }
}

bool AdamOptimizer::save(const char* filename) const {

    XMLDocument doc;
    XMLDeclaration* declaration = doc.NewDeclaration();
    doc.LinkEndChild(declaration);

    XMLElement* root = doc.NewElement("AdamOptimizer");
    doc.LinkEndChild(root);

    root->SetAttribute("beta1", beta1);
    root->SetAttribute("beta2", beta2);
    root->SetAttribute("mubeta1", mubeta1);
    root->SetAttribute("mubeta2", mubeta2);
    root->SetAttribute("verticesLearningRate", verticesLearningRate);
    root->SetAttribute("colorsLearningRate", colorsLearningRate);
    root->SetAttribute("muLearningRate", muLearningRate);
    root->SetAttribute("epsilon", epsilon);

    return doc.SaveFile(filename) == XML_SUCCESS;

    return true;
}

template <typename T>
void queryAttribute(XMLElement* element, const char* attributeName, T* value);

template <>
void queryAttribute<double>(XMLElement* element, const char* attributeName, double* value) {
    element->QueryDoubleAttribute(attributeName, value);
}

template <>
void queryAttribute<float>(XMLElement* element, const char* attributeName, float* value) {
    element->QueryFloatAttribute(attributeName, value);
}

bool AdamOptimizer::read(const char* filename) {
    XMLDocument doc;
    if (doc.LoadFile(filename) != XML_SUCCESS) {
        return false;
    }

    XMLElement* root = doc.RootElement();
    if (!root) {
        return false;
    }

    queryAttribute(root, "beta1", &beta1);
    queryAttribute(root, "beta2", &beta2);
    queryAttribute(root, "mubeta1", &mubeta1);
    queryAttribute(root, "mubeta2", &mubeta2);
    queryAttribute(root, "verticesLearningRate", &verticesLearningRate);
    queryAttribute(root, "colorsLearningRate", &colorsLearningRate);
    queryAttribute(root, "muLearningRate", &muLearningRate);
    queryAttribute(root, "epsilon", &epsilon);
    // reset t
    t = 0;
    return true;
}

bool TriangleMesh::save(const char* filename) const {

    XMLDocument doc;
    XMLDeclaration* declaration = doc.NewDeclaration();
    doc.LinkEndChild(declaration);

    XMLElement* root = doc.NewElement("TriangleMesh");
    doc.LinkEndChild(root);

    // Write bg
    XMLElement* bgElement = doc.NewElement("bgIndex");
    bgElement->SetAttribute("value", bg_index);
    root->LinkEndChild(bgElement);

    // 写入vertices
    XMLElement* verticesElement = doc.NewElement("Vertices");
    for (const auto& vertex : vertices) {
        XMLElement* vertexElement = doc.NewElement("Vertex");
        vertexElement->SetAttribute("x", vertex.x);
        vertexElement->SetAttribute("y", vertex.y);
        verticesElement->LinkEndChild(vertexElement);
    }
    root->LinkEndChild(verticesElement);

    // 写入indices
    XMLElement* indicesElement = doc.NewElement("Indices");
    for (const auto& index : indices) {
        XMLElement* indexElement = doc.NewElement("Index");
        indexElement->SetAttribute("v0", index.x);
        indexElement->SetAttribute("v1", index.y);
        indexElement->SetAttribute("v2", index.z);
        indicesElement->LinkEndChild(indexElement);
    }
    root->LinkEndChild(indicesElement);

    // 写入colors
    XMLElement* colorsElement = doc.NewElement("Colors");
    for (const auto& color : colors) {
        XMLElement* colorElement = doc.NewElement("Color");
        colorElement->SetAttribute("r", color.x);
        colorElement->SetAttribute("g", color.y);
        colorElement->SetAttribute("b", color.z);
        colorsElement->LinkEndChild(colorElement);
    }
    root->LinkEndChild(colorsElement);

    // 写入orders
    XMLElement* ordersElement = doc.NewElement("Orders");
    for (const auto& order : orders) {
        XMLElement* orderElement = doc.NewElement("Order");
        orderElement->SetAttribute("value", order);
        ordersElement->LinkEndChild(orderElement);
    }
    root->LinkEndChild(ordersElement);

    // 写入mu
    XMLElement* muElement = doc.NewElement("Mu");
    for (const auto& muValue : mu) {
        XMLElement* muValueElement = doc.NewElement("MuValue");
        muValueElement->SetAttribute("value", muValue);
        muElement->LinkEndChild(muValueElement);
    }
    root->LinkEndChild(muElement);

    // 保存到文件
    return doc.SaveFile(filename) == XML_SUCCESS;
}

bool TriangleMesh::read(const char* filename) {
 
    XMLDocument doc;
    if (doc.LoadFile(filename) != XML_SUCCESS) {
        return false; // 读取失败
    }

    XMLElement* root = doc.RootElement();
    if (!root) {
        return false; // 没有根元素
    }

    XMLElement* bgElement = root->FirstChildElement("bgIndex");
    if (bgElement) {
        bgElement->QueryIntAttribute("value", &bg_index);
    }

    // 读取vertices
    XMLElement* verticesElement = root->FirstChildElement("Vertices");
    if (verticesElement) {
        for (XMLElement* vertexElement = verticesElement->FirstChildElement("Vertex"); vertexElement; vertexElement = vertexElement->NextSiblingElement()) {
            Vec2f vertex;
            vertexElement->QueryFloatAttribute("x", &vertex.x);
            vertexElement->QueryFloatAttribute("y", &vertex.y);
            vertices.push_back(vertex);
        }
    }

    // 读取indices
    XMLElement* indicesElement = root->FirstChildElement("Indices");
    if (indicesElement) {
        for (XMLElement* indexElement = indicesElement->FirstChildElement("Index"); indexElement; indexElement = indexElement->NextSiblingElement()) {
            Vec3i index;
            indexElement->QueryIntAttribute("v0", &index.x);
            indexElement->QueryIntAttribute("v1", &index.y);
            indexElement->QueryIntAttribute("v2", &index.z);
            indices.push_back(index);
        }
    }

    // 读取colors
    XMLElement* colorsElement = root->FirstChildElement("Colors");
    if (colorsElement) {
        for (XMLElement* colorElement = colorsElement->FirstChildElement("Color"); colorElement; colorElement = colorElement->NextSiblingElement()) {
            Vec3f color;
            colorElement->QueryFloatAttribute("r", &color.x);
            colorElement->QueryFloatAttribute("g", &color.y);
            colorElement->QueryFloatAttribute("b", &color.z);
            colors.push_back(color);
        }
    }

    // 读取orders
    XMLElement* ordersElement = root->FirstChildElement("Orders");
    if (ordersElement) {
        for (XMLElement* orderElement = ordersElement->FirstChildElement("Order"); orderElement; orderElement = orderElement->NextSiblingElement()) {
            int order;
            orderElement->QueryIntAttribute("value", &order);
            orders.push_back(order);
        }
    }

    // 读取mu
    XMLElement* muElement = root->FirstChildElement("Mu");
    if (muElement) {
        for (XMLElement* muValueElement = muElement->FirstChildElement("MuValue"); muValueElement; muValueElement = muValueElement->NextSiblingElement()) {
            Real muValue;
            muValueElement->QueryFloatAttribute("value", &muValue);
            mu.push_back(muValue);
        }
    }

    return true;
}


Real cosine(Vec2f edge1, Vec2f edge2) {
    return dot(edge1, edge2) / (length(edge1) * length(edge2));
}

// Keep every triangle: 1. no overlap (128, 128) 2. the range of largest angle and smallest angle
// 3. the area <= 256*256/n
bool valid_triangle(Vec2f v0, Vec2f v1, Vec2f v2, int n) {

    Vec2f edge1to2(v1.x - v0.x, v1.y - v0.y);
    Vec2f edge2to3(v2.x - v1.x, v2.y - v1.y);
    Vec2f edge3to1(v0.x - v2.x, v0.y - v2.y);

    Real cos1 = cosine(-1 * edge1to2, edge2to3);
    Real cos2 = cosine(-1 * edge2to3, edge3to1);
    Real cos3 = cosine(-1 * edge3to1, edge1to2);

    Real minCos = std::min({ cos1, cos2, cos3 });

    if (minCos < -0.5) {
        return false;
    }

    Real maxCos = std::max({ cos1, cos2, cos3 });

    if (maxCos > 0.95) {
        return false;
    }

    // form three half-planes: v1-v0, v2-v1, v0-v2
    // if a point is on the same side of all three half-planes, it's inside the triangle.
    auto n01 = normal(v1 - v0), n12 = normal(v2 - v1), n20 = normal(v0 - v2);
    Vec2f screen_pos{ 128, 128 };
    auto side01 = dot(screen_pos - v0, n01) > 0;
    auto side12 = dot(screen_pos - v1, n12) > 0;
    auto side20 = dot(screen_pos - v2, n20) > 0;
    if ((side01 && side12 && side20) || (!side01 && !side12 && !side20)) {
        return false;
    }

    Real l1 = length(edge1to2);
    Real l2 = length(edge2to3);
    Real l3 = length(edge3to1);

    Real p = 0.5 * (l1 + l2 + l3);

    if (sqrt(p * (p - l1) * (p - l2) * (p - l3)) > 256 * 256 / n) {
        return false;
    }

    return true;

}


TriangleMesh::TriangleMesh(int n, std::mt19937& rng) {

    Real denom = (Real)pow(n, 2.0 / 3.0);
    // 生成 n 个三角形的相关数据
    for (int i = 0; i < n; ++i) {

        // 生成三个随机顶点
        Vec2f v1{ distribution(rng) * 256, distribution(rng) * 256 };
        Vec2f v2{ distribution(rng) * 256, distribution(rng) * 256 };
        Vec2f v3{ distribution(rng) * 256, distribution(rng) * 256 };

        while (!valid_triangle(v1, v2, v3, n)) {
            v1 = { distribution(rng) * 256, distribution(rng) * 256 };
            v2 = { distribution(rng) * 256, distribution(rng) * 256 };
            v3 = { distribution(rng) * 256, distribution(rng) * 256 };
        }

        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v3);

        // 生成对应的 indices
        indices.push_back(Vec3i{ i * 3, i * 3 + 1, i * 3 + 2 });

        // 生成随机颜色
        colors.push_back(Vec3f{ distribution(rng), distribution(rng), distribution(rng) });

        // 生成 orders 和 mu
        orders.push_back(i);
        mu.push_back(0.0);
    }
    mu.pop_back();
    bg_index = -1;
}

TriangleMesh::TriangleMesh(MESH_TYPE m, int n, mt19937& rng) {
    if (m == MESH_TYPE::INIT) {
        string filename = "initialization_" + to_string(n) + ".xml";

        if (!read(filename.c_str())) {
            *this = TriangleMesh(n, rng);
            save(filename.c_str());
        }
    }
    else if (m == MESH_TYPE::GOAL) {
        string filename = "target_" + to_string(n) + ".xml";
        string initFilename = "initialization_" + to_string(n) + ".xml";

        if (!read(filename.c_str())) {
            if (!read(initFilename.c_str())) {
                TriangleMesh init = TriangleMesh(n, rng);
                init.save(initFilename.c_str());
                *this = TriangleMesh(init, POS_REF_RATIO, COL_REF_RATIO, rng);
                save(filename.c_str());
            }
            else {
                TriangleMesh init(MESH_TYPE::INIT, n, rng);
                *this = TriangleMesh(init, POS_REF_RATIO, COL_REF_RATIO, rng);
                save(filename.c_str());
            }
        }
    }
}

TriangleMesh::TriangleMesh(const TriangleMesh& mesh, Real posRefRatio, Real colRefRatio, mt19937& rng) {

    int n = mesh.orders.size();
    for (int i = 0; i < n; ++i) {
        //生成三个随机顶点
        Vec2f v1{ distribution(rng) * 256, distribution(rng) * 256 };
        Vec2f v2{ distribution(rng) * 256, distribution(rng) * 256 };
        Vec2f v3{ distribution(rng) * 256, distribution(rng) * 256 };

        v1 = (1 - posRefRatio) * v1 + posRefRatio * mesh.vertices[i * 3];
        v2 = (1 - posRefRatio) * v2 + posRefRatio * mesh.vertices[i * 3 + 1];
        v3 = (1 - posRefRatio) * v3 + posRefRatio * mesh.vertices[i * 3 + 2];

        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v3);

        // 生成对应的 indices
        indices.push_back(Vec3i{ i * 3, i * 3 + 1, i * 3 + 2 });

        // 生成随机颜色
        Real t = distribution(rng);
        Vec3f random;
        random.x = ((int)(128 * t) % 2 == 0) ? t : 1 - t;
        random.y = ((int)(1024 * t) % 2 == 0) ? t : 1 - t;
        random.z = ((int)(32768 * t) % 2 == 0) ? t : 1 - t;
        colors.push_back(colRefRatio * mesh.colors[i] + (1 - colRefRatio) * random);

        // 生成 orders 和 mu
        orders.push_back(i);
        mu.push_back(0.0);
    }
    // 使用 std::shuffle 进行随机排列
    std::shuffle(orders.begin(), orders.end(), rng);
    mu.pop_back();
    bg_index = -1;
}

void create_video(const std::vector<Img>& frames, const std::string& output_filename, int fps) {
    cv::Size frame_size(frames[0].width, frames[0].height);
    cv::VideoWriter video(output_filename, cv::VideoWriter::fourcc('X', '2', '6', '4'), fps, frame_size);

    if (!video.isOpened()) {
        std::cerr << "Error opening video file." << std::endl;
        return;
    }

    auto tonemap = [](Real x) {
        return int(pow(clamp(x, Real(0), Real(1)), Real(1 / 2.2)) * 255 + Real(.5)); };

    for (const auto& frame : frames) {
        cv::Mat image(frame.height, frame.width, CV_8UC3);

        for (int y = 0; y < frame.height; ++y) {
            for (int x = 0; x < frame.width; ++x) {
                Vec3f pixel = frame.color[y * frame.width + x];
                image.at<cv::Vec3b>(y, x) = cv::Vec3b{
                    static_cast<uchar>(tonemap(pixel.x)),
                    static_cast<uchar>(tonemap(pixel.y)),
                    static_cast<uchar>(tonemap(pixel.z))
                };
            }
        }

        video.write(image);
    }

    video.release();
}

void TriangleMesh::bgCreate(int width, int height, const Vec3f& color, int order) {
    assert(bg_index == -1);
    int indices_num = vertices.size();

    assert(orders.size() == indices.size());
    bg_index = orders.size();
    if (order == -1) {
        orders.push_back(indices.size());
    }
    else {
        auto insertPos = orders.begin() + order;
        orders.insert(insertPos, indices.size());
    }

    mu.push_back(0.0);
    indices.push_back(Vec3i{ indices_num, indices_num + 1, indices_num + 2 });

    vertices.push_back(Vec2f{ -1, -1 });
    vertices.push_back(Vec2f{ 2 * (Real)width + 2, -1 });
    vertices.push_back(Vec2f{ -1, 2 * (Real)height + 2 });

    colors.push_back(color);
}




vector<int> kMeans(const vector<Vec3f>& data, int k, vector<Vec3f>& centroids) {
    const int numPoints = data.size();
    assert(k == centroids.size());
    vector<int> assignments(numPoints, 0);

    // Run the algorithm until convergence
    bool converged = false;
    int iter = 0;

    while (!converged) {
        // Assign each point to the nearest centroid
        for (int i = 0; i < numPoints; ++i) {
            Real minDist = numeric_limits<Real>::max();
            int minIndex = 0;

            for (int j = 0; j < k; ++j) {
                Real dist = squaredDistance(data[i], centroids[j]);
                if (dist < minDist) {
                    minDist = dist;
                    minIndex = j;
                }
            }

            assignments[i] = minIndex;
        }

        // Update the centroids based on the assigned points
        vector<int> counts(k, 0);
        vector<Vec3f> sums(k, Vec3f{ 0, 0, 0 });

        for (int i = 0; i < numPoints; ++i) {
            int clusterIndex = assignments[i];
            ++counts[clusterIndex];

            sums[clusterIndex] += data[i];
        }

        // Check for convergence
        Real maxChange = 0.0;
        for (int j = 0; j < k; ++j) {
            if (counts[j] > 0) {
                Vec3f oldCentroid = centroids[j];
                centroids[j] = sums[j] / counts[j];
                Real change = squaredDistance(centroids[j], oldCentroid);
                maxChange = max(maxChange, change);
            }
        }

        ++iter;

        // Check if converged
        converged = (maxChange < KMEANS_ERROR) || (iter >= 100);  // Added iteration limit as a safety measure
    }

    return assignments;
}

cv::Mat vectorToBinaryMat(const vector<int>& inputVector, int value, int width, int height) {

    assert(width * height == inputVector.size());

    cv::Mat binaryMat(height, width, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < inputVector.size(); ++i) {
        if (inputVector[i] == value) {
            int row = i / width;
            int col = i % width;

            binaryMat.at<uchar>(row, col) = 1;
        }
    }

    return binaryMat;
}

cv::Mat ImgToGrayMat(const Img& input, int channel) {

    assert(channel == 0 || channel == 1 || channel == 2);

    cv::Mat Mat(input.height, input.width, CV_8UC1, cv::Scalar(0));

    for (int i = 0; i < input.color.size(); ++i) {
        int row = i / input.width;
        int col = i % input.width;
        Mat.at<uchar>(row, col) = static_cast<uchar>(input.color[row * input.width + col][channel]);
    }

    return Mat;
}

vector<int> partitionColors(const Img& img, int k, const Vec3f& bg_color) {

    vector<pair<Vec3f, int>> color_index;
    for (int i = 0; i < img.color.size(); i++) {
        color_index.push_back({ img.color[i], i });
    }

    sort(color_index.begin(), color_index.end(), [](const pair<Vec3f, int>& a, const pair<Vec3f, int>& b) {
        return a.first < b.first;
    });


    int start = 0;
    for (; start < color_index.size(); start++) {
        if (!(color_index[start].first == bg_color))break;
    }

    int avg_count = (color_index.size() - start) / k;
    int remaining = (color_index.size() - start) % k;

    vector<int> assignments(img.width * img.height, 0);

    for (int i = 0; i < k; ++i) {
        int end = start + avg_count + (i < remaining ? 1 : 0);
        for (int j = start; j < end; j++) {
            // value == 0 means background
            assignments[color_index[j].second] = i + 1;
        }
        start = end;
    }

    return assignments;
}

TriangleMesh::TriangleMesh(int n, const Img& target, const Vec3f& bg_color, mt19937& rng) {
    // K-means for color
    vector<Vec3f> tmp_colors;
    for (int i = 0; i < n; ++i) {
        tmp_colors.push_back(Vec3f{ distribution(rng), distribution(rng), distribution(rng) });
    }
    vector<int> assignments = kMeans(target.color, n, tmp_colors);
    vector<vector<int>> labels(target.height, vector<int>(target.width, 0));
    int label = 0;
    vector<pair<int, int>> equal;
    for (int row = 0; row < target.height; row++) {
        for (int col = 0; col < target.width; col++) {
            int uplabel = -1;
            int leftlabel = -1;
            if (row > 0)uplabel = labels[row - 1][col];
            if (col > 0)leftlabel = labels[row][col - 1];

            int pos = row * target.width + col;
            // now assignments[pos] might be 0
            int upassign = -1;
            int leftassign = -1;
            int myassign = assignments[pos];
            if (row > 0)upassign = assignments[(row - 1) * target.width + col];
            if (col > 0)leftassign = assignments[row * target.width + col - 1];
            if (myassign != upassign && myassign != leftassign) {
                labels[row][col] = label;
                label++;
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

    UnionFind uf(label);
    for (pair<int, int> item : equal) {
        uf.unionSets(item.first, item.second);
    }

    vector<record> rec(label, record());
    for (int row = 0; row < target.height; row++) {
        for (int col = 0; col < target.width; col++) {
            int this_label = uf.find(labels[row][col]);
            rec[this_label].count++;
            rec[this_label].sum_col += col;
            rec[this_label].sum_row += row;
            rec[this_label].sum_color += target.color[row * target.width + col];
        }
    }

    // sort acoording to area
    sort(rec.begin(), rec.end(), [](const record& a, const record& b) {
        return b.count < a.count;
        });

    int count = 0;
    for (int i = 0; count < n && i < rec.size(); i++) {
        Vec3f tmpColor = rec[i].sum_color / (Real)rec[i].count;
        if (length(tmpColor - bg_color) < INI_ERROR) continue;
        // calculate the average color

        colors.push_back(tmpColor);

        Real r = sqrt(rec[i].count * 4.0 / 3.0 / sqrt(3));
        Real center_x = rec[i].sum_col / rec[i].count;
        Real center_y = rec[i].sum_row / rec[i].count;

        vertices.push_back(Vec2f{ (Real)(center_x + r * sqrt(3) / 2),(Real)(center_y - 0.5 * r) });
        vertices.push_back(Vec2f{ (Real)(center_x - r * sqrt(3) / 2),(Real)(center_y - 0.5 * r) });
        vertices.push_back(Vec2f{ center_x, (Real)(center_y + r) });

        mu.push_back(0.0);
        orders.push_back(count);

        indices.push_back(Vec3i{ 3 * count,3 * count + 1, 3 * count + 2 });
        count++;
    }

    shuffle(orders.begin(), orders.end(), rng);

    // mu should be n-1
    mu.pop_back();

    // add bg
    bg_index = -1;
    bgCreate(target.width, target.height, bg_color);
}