#pragma once

#include <vector>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"

using namespace std;

class Scene
{
private:
    ifstream fp_in;
    void loadFromJSON(const std::string& jsonName);
    bool loadGLTF(const std::string& gltfPath, const std::string& materialName, const glm::mat4& transform = glm::mat4(1.0f));
    void addTriangleFromGLTF(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                            const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
                            int materialId, const glm::mat4& transform = glm::mat4(1.0f));
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    RenderState state;
};
