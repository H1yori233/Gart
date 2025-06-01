#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1;
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];
        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
        }
        else if (type == "triangle")
        {
            newGeom.type = TRIANGLE;
            
            // Parse triangle vertices
            const auto& v0 = p["V0"];
            const auto& v1 = p["V1"];
            const auto& v2 = p["V2"];
            newGeom.triangle.v0 = glm::vec3(v0[0], v0[1], v0[2]);
            newGeom.triangle.v1 = glm::vec3(v1[0], v1[1], v1[2]);
            newGeom.triangle.v2 = glm::vec3(v2[0], v2[1], v2[2]);
            
            // Parse vertex normals (optional)
            if (p.contains("N0") && p.contains("N1") && p.contains("N2"))
            {
                const auto& n0 = p["N0"];
                const auto& n1 = p["N1"];
                const auto& n2 = p["N2"];
                newGeom.triangle.n0 = glm::normalize(glm::vec3(n0[0], n0[1], n0[2]));
                newGeom.triangle.n1 = glm::normalize(glm::vec3(n1[0], n1[1], n1[2]));
                newGeom.triangle.n2 = glm::normalize(glm::vec3(n2[0], n2[1], n2[2]));
                newGeom.triangle.hasVertexNormals = true;
            }
            else
            {
                // Use default normals (will be computed from face normal)
                newGeom.triangle.hasVertexNormals = false;
            }
            
            // Parse texture coordinates (optional)
            if (p.contains("T0") && p.contains("T1") && p.contains("T2"))
            {
                const auto& t0 = p["T0"];
                const auto& t1 = p["T1"];
                const auto& t2 = p["T2"];
                newGeom.triangle.t0 = glm::vec2(t0[0], t0[1]);
                newGeom.triangle.t1 = glm::vec2(t1[0], t1[1]);
                newGeom.triangle.t2 = glm::vec2(t2[0], t2[1]);
            }
            else
            {
                // Default texture coordinates
                newGeom.triangle.t0 = glm::vec2(0.0f, 0.0f);
                newGeom.triangle.t1 = glm::vec2(1.0f, 0.0f);
                newGeom.triangle.t2 = glm::vec2(0.5f, 1.0f);
            }
        }
        else
        {
            newGeom.type = SPHERE;
        }
        
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        
        // For triangles, transformation is applied directly to vertices, 
        // but we still need to set up the transformation matrices for consistency
        if (newGeom.type == TRIANGLE)
        {
            // For triangles, we can use identity matrices since vertices are in world space
            // Or apply transformation if TRANS/ROTAT/SCALE are provided
            if (p.contains("TRANS") && p.contains("ROTAT") && p.contains("SCALE"))
            {
                const auto& trans = p["TRANS"];
                const auto& rotat = p["ROTAT"];
                const auto& scale = p["SCALE"];
                newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
                newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
                newGeom.transform = utilityCore::buildTransformationMatrix(
                    newGeom.translation, newGeom.rotation, newGeom.scale);
                newGeom.inverseTransform = glm::inverse(newGeom.transform);
                newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
            }
            else
            {
                // Identity matrices
                newGeom.translation = glm::vec3(0.0f);
                newGeom.rotation = glm::vec3(0.0f);
                newGeom.scale = glm::vec3(1.0f);
                newGeom.transform = glm::mat4(1.0f);
                newGeom.inverseTransform = glm::mat4(1.0f);
                newGeom.invTranspose = glm::mat4(1.0f);
            }
        }
        else
        {
            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];
            newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            newGeom.transform = utilityCore::buildTransformationMatrix(
                newGeom.translation, newGeom.rotation, newGeom.scale);
            newGeom.inverseTransform = glm::inverse(newGeom.transform);
            newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
        }

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
