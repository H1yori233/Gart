#include <iostream>
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <unordered_map>
#include "json.hpp"

#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE_IMPLEMENTATION
#include "gltf/tiny_gltf.h"
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

void Scene::loadFromJSON(const std::string &jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto &materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto &item : materialsData.items())
    {
        const auto &name = item.key();
        const auto &p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto &col = p["RGB"];
            newMaterial.color = make_constant_spectrum_texture(glm::vec3(col[0], col[1], col[2]));
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto &col = p["RGB"];
            newMaterial.color = make_constant_spectrum_texture(glm::vec3(col[0], col[1], col[2]));
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto &col = p["RGB"];
            newMaterial.color = make_constant_spectrum_texture(glm::vec3(col[0], col[1], col[2]));
            newMaterial.hasReflective = 1.0f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Dielectric")
        {
            const auto &col = p["RGB"];
            newMaterial.color = make_constant_spectrum_texture(glm::vec3(col[0], col[1], col[2]));
            newMaterial.hasRefractive = 1.0f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto &objectsData = data["Objects"];
    for (const auto &p : objectsData)
    {
        const auto &type = p["TYPE"];
        if (type == "mesh")
        {
            std::string meshPath = p["FILE"];
            auto meshExt = meshPath.substr(meshPath.find_last_of('.'));
            Geom baseGeom;
            baseGeom.materialid = MatNameToID[p["MATERIAL"]];

            const auto &trans = p["TRANS"];
            const auto &rotat = p["ROTAT"];
            const auto &scale = p["SCALE"];
            baseGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
            baseGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
            baseGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
            baseGeom.transform = utilityCore::buildTransformationMatrix(
                baseGeom.translation, baseGeom.rotation, baseGeom.scale);
            baseGeom.inverseTransform = glm::inverse(baseGeom.transform);
            baseGeom.invTranspose = glm::inverseTranspose(baseGeom.transform);

            if (meshExt == ".obj")
            {
                // loadMeshFromOBJ(meshPath, baseGeom);
            }
            else if (meshExt == ".gltf" || meshExt == ".glb")
            {
                loadMeshFromGLTF(meshPath, baseGeom);
            }
            else
            {
                std::cerr << "Unsupported Mesh: " << meshExt << std::endl;
            }

            continue;
        }

        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto &trans = p["TRANS"];
        const auto &rotat = p["ROTAT"];
        const auto &scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
        if (materials[newGeom.materialid].emittance > 0.0f) 
        {
            emitters.push_back(newGeom);
        }
    }
    const auto &cameraData = data["Camera"];
    Camera &camera = state.camera;
    RenderState &state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto &pos = cameraData["EYE"];
    const auto &lookat = cameraData["LOOKAT"];
    const auto &up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    if (cameraData.find("LENS_RADIUS") != cameraData.end()) 
    {
        camera.lensRadius = cameraData["LENS_RADIUS"];
    } 
    else 
    {
        camera.lensRadius = 0.f;
    }

    if (cameraData.find("FOCAL_DISTANCE") != cameraData.end()) 
    {
        camera.focalDistance = cameraData["FOCAL_DISTANCE"];
    } 
    else 
    {
        camera.focalDistance = 0.f;
    }

    // calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
                                   2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    // set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}

void Scene::loadFromGLTF(const std::string &gltfName)
{
    std::cout << "Begin GLTF Loading Mesh from: " << gltfName << std::endl;

    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret = false;
    if (gltfName.find(".glb") != std::string::npos)
    {
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, gltfName);
    }
    else
    {
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, gltfName);
    }

    if (!warn.empty())
    {
        std::cout << "glTF Load Warn: " << warn << std::endl;
    }

    if (!err.empty())
    {
        std::cerr << "glTF Load Error: " << err << std::endl;
    }

    if (!ret)
    {
        std::cerr << "Can not load glTF file: " << gltfName << std::endl;
        return;
    }

    Material newMaterial{};
    newMaterial.color = make_constant_spectrum_texture(glm::vec3(0.8f, 0.8f, 0.8f));
    uint32_t materialId = this->materials.size();
    this->materials.push_back(newMaterial);

    for (const auto &mesh : model.meshes)
    {
        for (const auto &primitive : mesh.primitives)
        {
            // Index
            const tinygltf::Accessor &indexAccessor = model.accessors[primitive.indices];
            const tinygltf::BufferView &indexBufferView = model.bufferViews[indexAccessor.bufferView];
            const tinygltf::Buffer &indexBuffer = model.buffers[indexBufferView.buffer];

            // Position
            const tinygltf::Accessor &posAccessor = model.accessors[primitive.attributes.find("POSITION")->second];
            const tinygltf::BufferView &posBufferView = model.bufferViews[posAccessor.bufferView];
            const tinygltf::Buffer &posBuffer = model.buffers[posBufferView.buffer];

            // Normal
            bool hasNormals = primitive.attributes.find("NORMAL") != primitive.attributes.end();
            tinygltf::Accessor normalAccessor;
            tinygltf::BufferView normalBufferView;
            tinygltf::Buffer normalBuffer;

            if (hasNormals)
            {
                normalAccessor = model.accessors[primitive.attributes.find("NORMAL")->second];
                normalBufferView = model.bufferViews[normalAccessor.bufferView];
                normalBuffer = model.buffers[normalBufferView.buffer];
            }

            // UV
            bool hasTexCoords = primitive.attributes.find("TEXCOORD_0") != primitive.attributes.end();
            tinygltf::Accessor texcoordAccessor;
            tinygltf::BufferView texcoordBufferView;
            tinygltf::Buffer texcoordBuffer;

            if (hasTexCoords)
            {
                texcoordAccessor = model.accessors[primitive.attributes.find("TEXCOORD_0")->second];
                texcoordBufferView = model.bufferViews[texcoordAccessor.bufferView];
                texcoordBuffer = model.buffers[texcoordBufferView.buffer];
            }

            size_t indexCount = indexAccessor.count;
            const unsigned char *indexData = &indexBuffer.data[indexBufferView.byteOffset + indexAccessor.byteOffset];
            for (size_t i = 0; i < indexCount; i += 3)
            {
                Geom geom;
                geom.type = TRIANGLE;
                geom.materialid = materialId;

                // Indices
                uint32_t indices[3];
                if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT)
                {
                    uint16_t *indexPtr = (uint16_t *)indexData;
                    indices[0] = indexPtr[i];
                    indices[1] = indexPtr[i + 1];
                    indices[2] = indexPtr[i + 2];
                }
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT)
                {
                    uint32_t *indexPtr = (uint32_t *)indexData;
                    indices[0] = indexPtr[i];
                    indices[1] = indexPtr[i + 1];
                    indices[2] = indexPtr[i + 2];
                }
                else if (indexAccessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE)
                {
                    uint8_t *indexPtr = (uint8_t *)indexData;
                    indices[0] = indexPtr[i];
                    indices[1] = indexPtr[i + 1];
                    indices[2] = indexPtr[i + 2];
                }

                // Positions
                const unsigned char *posData = &posBuffer.data[posBufferView.byteOffset + posAccessor.byteOffset];
                size_t posStride = posAccessor.ByteStride(posBufferView) ? posAccessor.ByteStride(posBufferView) : sizeof(float) * 3;

                float *v0 = (float *)(posData + indices[0] * posStride);
                float *v1 = (float *)(posData + indices[1] * posStride);
                float *v2 = (float *)(posData + indices[2] * posStride);

                geom.triangle.v0 = glm::vec3(v0[0], v0[1], v0[2]);
                geom.triangle.v1 = glm::vec3(v1[0], v1[1], v1[2]);
                geom.triangle.v2 = glm::vec3(v2[0], v2[1], v2[2]);

                // Normals
                if (hasNormals)
                {
                    const unsigned char *normalData = &normalBuffer.data[normalBufferView.byteOffset + normalAccessor.byteOffset];
                    size_t normalStride = normalAccessor.ByteStride(normalBufferView) ? normalAccessor.ByteStride(normalBufferView) : sizeof(float) * 3;

                    float *n0 = (float *)(normalData + indices[0] * normalStride);
                    float *n1 = (float *)(normalData + indices[1] * normalStride);
                    float *n2 = (float *)(normalData + indices[2] * normalStride);

                    geom.triangle.n0 = glm::vec3(n0[0], n0[1], n0[2]);
                    geom.triangle.n1 = glm::vec3(n1[0], n1[1], n1[2]);
                    geom.triangle.n2 = glm::vec3(n2[0], n2[1], n2[2]);
                }
                else
                {
                    glm::vec3 normal = glm::normalize(glm::cross(
                        geom.triangle.v1 - geom.triangle.v0,
                        geom.triangle.v2 - geom.triangle.v0));
                    geom.triangle.n0 = normal;
                    geom.triangle.n1 = normal;
                    geom.triangle.n2 = normal;
                }

                // UVs
                if (hasTexCoords)
                {
                    const unsigned char *texcoordData = &texcoordBuffer.data[texcoordBufferView.byteOffset + texcoordAccessor.byteOffset];
                    size_t texcoordStride = texcoordAccessor.ByteStride(texcoordBufferView) ? texcoordAccessor.ByteStride(texcoordBufferView) : sizeof(float) * 2;

                    float *t0 = (float *)(texcoordData + indices[0] * texcoordStride);
                    float *t1 = (float *)(texcoordData + indices[1] * texcoordStride);
                    float *t2 = (float *)(texcoordData + indices[2] * texcoordStride);

                    geom.triangle.t0 = glm::vec2(t0[0], t0[1]);
                    geom.triangle.t1 = glm::vec2(t1[0], t1[1]);
                    geom.triangle.t2 = glm::vec2(t2[0], t2[1]);
                }
                else
                {
                    geom.triangle.t0 = glm::vec2(0.0f, 0.0f);
                    geom.triangle.t1 = glm::vec2(1.0f, 0.0f);
                    geom.triangle.t2 = glm::vec2(0.0f, 1.0f);
                }

                geom.translation = glm::vec3(0.0f);
                geom.rotation = glm::vec3(0.0f);
                geom.scale = glm::vec3(1.0f);
                geom.transform = glm::mat4(1.0f);
                geom.inverseTransform = glm::mat4(1.0f);
                geom.invTranspose = glm::mat4(1.0f);

                geoms.push_back(geom);
                if (materials[geom.materialid].emittance > 0.0f) 
                {
                    emitters.push_back(geom);
                }
            }
        }
    }

    std::cout << "Load GLTF Mesh End: " << gltfName << std::endl;
}

void Scene::loadMeshFromGLTF(const std::string &gltfPath, const Geom &baseGeom)
{
    size_t offset = geoms.size();
    loadFromGLTF(gltfPath);

    std::cout << "Begin Set GLTF Geoms from: " << gltfPath << std::endl;

    for (size_t i = offset; i < geoms.size(); i++)
    {
        geoms[i].materialid = baseGeom.materialid;

        // Transform
        glm::vec4 v0 = baseGeom.transform * glm::vec4(geoms[i].triangle.v0, 1.0f);
        glm::vec4 v1 = baseGeom.transform * glm::vec4(geoms[i].triangle.v1, 1.0f);
        glm::vec4 v2 = baseGeom.transform * glm::vec4(geoms[i].triangle.v2, 1.0f);
        geoms[i].triangle.v0 = glm::vec3(v0) / v0.w;
        geoms[i].triangle.v1 = glm::vec3(v1) / v1.w;
        geoms[i].triangle.v2 = glm::vec3(v2) / v2.w;

        // Normals
        glm::vec4 n0 = baseGeom.invTranspose * glm::vec4(geoms[i].triangle.n0, 0.0f);
        glm::vec4 n1 = baseGeom.invTranspose * glm::vec4(geoms[i].triangle.n1, 0.0f);
        glm::vec4 n2 = baseGeom.invTranspose * glm::vec4(geoms[i].triangle.n2, 0.0f);
        geoms[i].triangle.n0 = glm::normalize(glm::vec3(n0));
        geoms[i].triangle.n1 = glm::normalize(glm::vec3(n1));
        geoms[i].triangle.n2 = glm::normalize(glm::vec3(n2));

        geoms[i].translation = baseGeom.translation;
        geoms[i].rotation = baseGeom.rotation;
        geoms[i].scale = baseGeom.scale;
        geoms[i].transform = baseGeom.transform;
        geoms[i].inverseTransform = baseGeom.inverseTransform;
        geoms[i].invTranspose = baseGeom.invTranspose;
    }

    std::cout << "Set GLTF Geoms End: " << gltfPath << std::endl;
    std::cout << "Number of Geoms: " << geoms.size() << std::endl;
}
