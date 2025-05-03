#pragma once

#include <glm/glm.hpp>
#include <string>
#include <stb_image.h>
#include <iostream>
#include <algorithm>
#include <cctype>
// #include <tinyexr.h>

using namespace std;

template <typename T>
class Image
{
private:
    int xSize;
    int ySize;
    T *pixels;

public:
    Image(int x, int y);
    ~Image();
    void setPixel(int x, int y, const T &pixel);
    T getPixel(int x, int y) const;
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);
};

using Image1 = Image<float>;
using Image3 = Image<glm::vec3>;

// 图像读取函数
// Image1 imread1(const std::filesystem::path &filename);
// Image3 imread3(const std::filesystem::path &filename);

// 辅助函数
inline std::string to_lowercase(const std::string &str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(), ::tolower);
    return result;
}

// inline void Error(const std::string &msg) {
//     std::cerr << "Error: " << msg << std::endl;
//     exit(1);
// }
