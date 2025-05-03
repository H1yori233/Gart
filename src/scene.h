#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
// #include "texture.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    void loadFromGLTF(const std::string& gltfName);
    void loadMeshFromGLTF(const std::string& gltfPath, const Geom& baseGeom);

public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Geom> emitters;
    std::vector<Material> materials;
    RenderState state;
};
