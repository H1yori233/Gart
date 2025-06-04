#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include <stb_image.h>
#include <stb_image_write.h>
#define TINYGLTF_IMPLEMENTATION
#include <tiny_gltf.h>
#include "json.hpp"
#include "scene.h"
using json = nlohmann::json;

__host__ __device__ inline float luminance(const glm::vec3& c) {
    return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z;
}

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        cout << "geoms size: " << geoms.size() << endl;

        buildLights();
        cout << "lightInfos size: " << lightInfos.size() << 
                ", totalLightPower: " << totalLightPower << endl;
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

IntegratorType stringToIntegratorType(const std::string& str) {
    for (int i = 0; i < INTEGRATOR_COUNT; i++) {
        if (str == INTEGRATOR_NAMES[i]) {
            return static_cast<IntegratorType>(i);
        }
    }
    std::cout << "Warning: Unknown integrator type '" << str << "', using default (nee)" << std::endl;
    return INTEGRATOR_NEE;
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
        else if (type == "quad")
        {
            newGeom.type = QUAD;
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
        else if (type == "gltf")
        {
            // Load GLTF model
            std::string gltfPath = p["PATH"];
            std::string materialName = p["MATERIAL"];
            
            // Calculate transformation matrix if TRANS/ROTAT/SCALE are provided
            glm::mat4 transform = glm::mat4(1.0f);
            if (p.contains("TRANS") && p.contains("ROTAT") && p.contains("SCALE"))
            {
                const auto& trans = p["TRANS"];
                const auto& rotat = p["ROTAT"];
                const auto& scale = p["SCALE"];
                glm::vec3 translation = glm::vec3(trans[0], trans[1], trans[2]);
                glm::vec3 rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
                glm::vec3 scaling = glm::vec3(scale[0], scale[1], scale[2]);
                transform = utilityCore::buildTransformationMatrix(translation, rotation, scaling);
            }
            
            if (loadGLTF(gltfPath, materialName, transform)) {
                std::cout << "Successfully loaded GLTF model: " << gltfPath << std::endl;
                continue; // Skip the normal geometry processing since GLTF loading adds triangles directly
            } else {
                std::cout << "Failed to load GLTF model: " << gltfPath << std::endl;
                continue;
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

    if (cameraData.contains("INTEGRATOR")) {
        std::string integratorStr = cameraData["INTEGRATOR"];
        state.integrator = stringToIntegratorType(integratorStr);
    } else {
        state.integrator = INTEGRATOR_NEE; // default
    }
    std::cout << "Using integrator: " << INTEGRATOR_NAMES[state.integrator] << std::endl;
    
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

void Scene::addTriangleFromGLTF(const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                               const glm::vec3& n0, const glm::vec3& n1, const glm::vec3& n2,
                               int materialId, const glm::mat4& transform)
{
    Geom newGeom;
    newGeom.type = TRIANGLE;
    newGeom.materialid = materialId;
    
    // Apply transformation to vertices
    glm::vec4 tv0 = transform * glm::vec4(v0, 1.0f);
    glm::vec4 tv1 = transform * glm::vec4(v1, 1.0f);
    glm::vec4 tv2 = transform * glm::vec4(v2, 1.0f);
    
    newGeom.triangle.v0 = glm::vec3(tv0);
    newGeom.triangle.v1 = glm::vec3(tv1);
    newGeom.triangle.v2 = glm::vec3(tv2);
    
    // Apply transformation to normals (using inverse transpose)
    glm::mat4 normalMatrix = glm::inverseTranspose(transform);
    glm::vec4 tn0 = normalMatrix * glm::vec4(n0, 0.0f);
    glm::vec4 tn1 = normalMatrix * glm::vec4(n1, 0.0f);
    glm::vec4 tn2 = normalMatrix * glm::vec4(n2, 0.0f);
    
    newGeom.triangle.n0 = glm::normalize(glm::vec3(tn0));
    newGeom.triangle.n1 = glm::normalize(glm::vec3(tn1));
    newGeom.triangle.n2 = glm::normalize(glm::vec3(tn2));
    newGeom.triangle.hasVertexNormals = true;
    
    // Default texture coordinates
    newGeom.triangle.t0 = glm::vec2(0.0f, 0.0f);
    newGeom.triangle.t1 = glm::vec2(1.0f, 0.0f);
    newGeom.triangle.t2 = glm::vec2(0.5f, 1.0f);
    
    // Set up transformation matrices
    newGeom.translation = glm::vec3(0.0f);
    newGeom.rotation = glm::vec3(0.0f);
    newGeom.scale = glm::vec3(1.0f);
    newGeom.transform = glm::mat4(1.0f);
    newGeom.inverseTransform = glm::mat4(1.0f);
    newGeom.invTranspose = glm::mat4(1.0f);
    
    geoms.push_back(newGeom);
}

bool Scene::loadGLTF(const std::string& gltfPath, const std::string& materialName, const glm::mat4& transform)
{
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;
    
    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfPath);
    
    if (!warn.empty()) {
        std::cout << "GLTF Warning: " << warn << std::endl;
    }
    
    if (!err.empty()) {
        std::cout << "GLTF Error: " << err << std::endl;
        return false;
    }
    
    if (!ret) {
        std::cout << "Failed to parse GLTF file: " << gltfPath << std::endl;
        return false;
    }
    
    // Find material ID by name
    int materialId = -1;
    for (size_t i = 0; i < materials.size(); ++i) {
        // We need to store material names somehow, for now use index 0 as default
        if (i == 0) { // Temporary: use first material
            materialId = i;
            break;
        }
    }
    
    if (materialId == -1) {
        std::cout << "Material '" << materialName << "' not found, using default" << std::endl;
        materialId = 0; // Use first material as fallback
    }
    
    // Process all meshes
    for (const auto& mesh : model.meshes) {
        for (const auto& primitive : mesh.primitives) {
            if (primitive.mode != TINYGLTF_MODE_TRIANGLES) {
                std::cout << "Skipping non-triangle primitive" << std::endl;
                continue;
            }
            
            // Get position accessor
            auto positionIt = primitive.attributes.find("POSITION");
            if (positionIt == primitive.attributes.end()) {
                std::cout << "No POSITION attribute found" << std::endl;
                continue;
            }
            
            // Get normal accessor
            auto normalIt = primitive.attributes.find("NORMAL");
            bool hasNormals = (normalIt != primitive.attributes.end());
            
            const tinygltf::Accessor& posAccessor = model.accessors[positionIt->second];
            const tinygltf::BufferView& posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer& posBuffer = model.buffers[posBufferView.buffer];
            
            const float* positions = reinterpret_cast<const float*>(&posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset]);
            
            const float* normals = nullptr;
            if (hasNormals) {
                const tinygltf::Accessor& normalAccessor = model.accessors[normalIt->second];
                const tinygltf::BufferView& normalBufferView = model.bufferViews[normalAccessor.bufferView];
                const tinygltf::Buffer& normalBuffer = model.buffers[normalBufferView.buffer];
                normals = reinterpret_cast<const float*>(&normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset]);
            }
            
            // Get indices if available
            if (primitive.indices >= 0) {
                const tinygltf::Accessor& indexAccessor = model.accessors[primitive.indices];
                const tinygltf::BufferView& indexBufferView = model.bufferViews[indexAccessor.bufferView];
                const tinygltf::Buffer& indexBuffer = model.buffers[indexBufferView.buffer];
                
                // Process triangles using indices
                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                    const uint16_t* indices = reinterpret_cast<const uint16_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                    
                    for (size_t i = 0; i < indexAccessor.count; i += 3) {
                        uint16_t i0 = indices[i];
                        uint16_t i1 = indices[i + 1];
                        uint16_t i2 = indices[i + 2];
                        
                        glm::vec3 v0(positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]);
                        glm::vec3 v1(positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]);
                        glm::vec3 v2(positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]);
                        
                        glm::vec3 n0, n1, n2;
                        if (hasNormals) {
                            n0 = glm::vec3(normals[i0 * 3], normals[i0 * 3 + 1], normals[i0 * 3 + 2]);
                            n1 = glm::vec3(normals[i1 * 3], normals[i1 * 3 + 1], normals[i1 * 3 + 2]);
                            n2 = glm::vec3(normals[i2 * 3], normals[i2 * 3 + 1], normals[i2 * 3 + 2]);
                        } else {
                            // Calculate face normal
                            glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                            n0 = n1 = n2 = faceNormal;
                        }
                        
                        addTriangleFromGLTF(v0, v1, v2, n0, n1, n2, materialId, transform);
                    }
                }
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                    const uint32_t* indices = reinterpret_cast<const uint32_t*>(&indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset]);
                    
                    for (size_t i = 0; i < indexAccessor.count; i += 3) {
                        uint32_t i0 = indices[i];
                        uint32_t i1 = indices[i + 1];
                        uint32_t i2 = indices[i + 2];
                        
                        glm::vec3 v0(positions[i0 * 3], positions[i0 * 3 + 1], positions[i0 * 3 + 2]);
                        glm::vec3 v1(positions[i1 * 3], positions[i1 * 3 + 1], positions[i1 * 3 + 2]);
                        glm::vec3 v2(positions[i2 * 3], positions[i2 * 3 + 1], positions[i2 * 3 + 2]);
                        
                        glm::vec3 n0, n1, n2;
                        if (hasNormals) {
                            n0 = glm::vec3(normals[i0 * 3], normals[i0 * 3 + 1], normals[i0 * 3 + 2]);
                            n1 = glm::vec3(normals[i1 * 3], normals[i1 * 3 + 1], normals[i1 * 3 + 2]);
                            n2 = glm::vec3(normals[i2 * 3], normals[i2 * 3 + 1], normals[i2 * 3 + 2]);
                        } else {
                            // Calculate face normal
                            glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                            n0 = n1 = n2 = faceNormal;
                        }
                        
                        addTriangleFromGLTF(v0, v1, v2, n0, n1, n2, materialId, transform);
                    }
                }
            }
            else {
                // No indices, process vertices directly
                for (size_t i = 0; i < posAccessor.count; i += 3) {
                    glm::vec3 v0(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
                    glm::vec3 v1(positions[(i + 1) * 3], positions[(i + 1) * 3 + 1], positions[(i + 1) * 3 + 2]);
                    glm::vec3 v2(positions[(i + 2) * 3], positions[(i + 2) * 3 + 1], positions[(i + 2) * 3 + 2]);
                    
                    glm::vec3 n0, n1, n2;
                    if (hasNormals) {
                        n0 = glm::vec3(normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]);
                        n1 = glm::vec3(normals[(i + 1) * 3], normals[(i + 1) * 3 + 1], normals[(i + 1) * 3 + 2]);
                        n2 = glm::vec3(normals[(i + 2) * 3], normals[(i + 2) * 3 + 1], normals[(i + 2) * 3 + 2]);
                    } else {
                        // Calculate face normal
                        glm::vec3 faceNormal = glm::normalize(glm::cross(v1 - v0, v2 - v0));
                        n0 = n1 = n2 = faceNormal;
                    }
                    
                    addTriangleFromGLTF(v0, v1, v2, n0, n1, n2, materialId, transform);
                }
            }
        }
    }
    
    std::cout << "Successfully loaded GLTF: " << gltfPath << " with " << model.meshes.size() << " meshes" << std::endl;
    return true;
}

void Scene::buildLights() {
    lightInfos.clear();
    totalLightPower = 0.0f;
    for (size_t i = 0; i < geoms.size(); ++i) {
        const Material& mat = materials[geoms[i].materialid];
        if (mat.emittance > 0.0f) {
            LightInfo lightInfo;
            lightInfo.geomid = i;
            lightInfo.emission = mat.color * mat.emittance;
            lightInfo.area = calculateGeomArea(geoms[i]);
            float luminance_weight = luminance(lightInfo.emission);
            lightInfo.power = luminance_weight * lightInfo.area;

            lightInfos.push_back(lightInfo);
            totalLightPower += lightInfo.power;
            cout << "Found light: geom_id=" << i 
                 << ", emission=(" << lightInfo.emission.x << "," << lightInfo.emission.y << "," << lightInfo.emission.z << ")"
                 << ", area=" << lightInfo.area 
                 << ", power=" << lightInfo.power << endl;
        }
    }
}

float Scene::calculateGeomArea(const Geom& geom) const {
    switch (geom.type) {
        case SPHERE: {
            glm::vec3 scale_vec = glm::vec3(
                glm::length(glm::vec3(geom.transform[0])),
                glm::length(glm::vec3(geom.transform[1])),
                glm::length(glm::vec3(geom.transform[2]))
            );
            float radius = (scale_vec.x + scale_vec.y + scale_vec.z) / 3.0f; // avg
            return 4 * PI * radius * radius;
        }

        case CUBE: {
            glm::vec3 scale_vec = glm::vec3(
                glm::length(glm::vec3(geom.transform[0])),
                glm::length(glm::vec3(geom.transform[1])),
                glm::length(glm::vec3(geom.transform[2]))
            );

            // plane light or not?
            float min_dim = glm::min(scale_vec.x, glm::min(scale_vec.y, scale_vec.z));
            float max_dim = glm::max(scale_vec.x, glm::max(scale_vec.y, scale_vec.z));
            float ratio = min_dim / max_dim;

            if (ratio < 0.1f) {
                if (scale_vec.x == min_dim) {
                    return scale_vec.y * scale_vec.z;
                } else if (scale_vec.y == min_dim) {
                    return scale_vec.x * scale_vec.z;
                } else {
                    return scale_vec.x * scale_vec.y;
                }
            } else {
                float total_area = 2.0f * (scale_vec.x * scale_vec.y + 
                                           scale_vec.y * scale_vec.z + 
                                           scale_vec.x * scale_vec.z);
                return total_area;
            }
        }
        
        case QUAD: {
            // Unit quad area (1x1) scaled by transform
            glm::vec3 edge1 = glm::vec3(geom.transform * glm::vec4(1, 0, 0, 0)); // X direction
            glm::vec3 edge2 = glm::vec3(geom.transform * glm::vec4(0, 1, 0, 0)); // Y direction
            return glm::length(glm::cross(edge1, edge2));
        }

        case TRIANGLE: {
            glm::vec3 v0 = geom.triangle.v0;
            glm::vec3 v1 = geom.triangle.v1;
            glm::vec3 v2 = geom.triangle.v2;
            glm::vec3 edge1 = v1 - v0;
            glm::vec3 edge2 = v2 - v0;
            return 0.5f * glm::length(glm::cross(edge1, edge2));
        }

        default:
            cout << "Warning: Unknown geometry type for area calculation, using default area 1.0" << endl;
            return 1.0f;
    }
}
