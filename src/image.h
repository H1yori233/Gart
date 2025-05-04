#pragma once

#include <glm/glm.hpp>
#include <string>
#include <stb_image.h>
#include <iostream>
#include <algorithm>
#include <cctype>
// #include <tinyexr.h>

#ifdef __CUDACC__
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

using namespace std;

template <typename T>
class Image
{
private:
    int xSize;
    int ySize;
    T *pixels;

public:
    Image();
    Image(int x, int y);
    ~Image();
    void setPixel(int x, int y, const T &pixel);
    T getPixel(int x, int y) const;
    void savePNG(const std::string &baseFilename);
    void saveHDR(const std::string &baseFilename);

    HOST_DEVICE inline int getWidth() const
    {
        return xSize;
    }

    HOST_DEVICE inline int getHeight() const
    {
        return ySize;
    }
};

using Image1 = Image<float>;
using Image3 = Image<glm::vec3>;

Image1 imread1(const std::string &filename);
Image3 imread3(const std::string &filename);

inline void Error(const std::string &msg) {
    std::cerr << "Error: " << msg << std::endl;
    exit(1);
}

HOST_DEVICE inline int modulo(int a, int b) {
    auto r = a % b;
    return (r < 0) ? r+b : r;
}

HOST_DEVICE inline float modulo(float a, float b) {
    float r = ::fmodf(a, b);
    return (r < 0.0f) ? r+b : r;
}

HOST_DEVICE inline double modulo(double a, double b) {
    double r = ::fmod(a, b);
    return (r < 0.0) ? r+b : r;
}
