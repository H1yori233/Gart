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

void Scene::loadFromJSON(const std::string &jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);

    // Load textures if present
    std::unordered_map<std::string, uint32_t> TexNameToID;
    std::vector<std::shared_ptr<TextureSpectrum>> textures;
    
    if (data.contains("Textures")) {
        const auto &texturesData = data["Textures"];
        for (const auto &item : texturesData.items()) {
            const std::string name = item.key();
            const auto &p = item.value();
            if (!p.contains("TYPE")) {
                std::cerr << "Texture \"" << name << "\" missing TYPE field." << std::endl;
                continue;
            }
            std::shared_ptr<TextureSpectrum> tex;
            std::string type = p["TYPE"];
            
            if (type == "constant") {
                auto col = p["RGB"];
                if (col.size() != 3) {
                    std::cerr << "Texture \"" << name << "\": constant missing RGB." << std::endl;
                    continue;
                }
                tex = std::make_shared<TextureSpectrum>(
                    make_constant_spectrum_texture(glm::vec3(col[0], col[1], col[2])));
            }
            else if (type == "checker") {
                auto c1 = p["COLOR1"], c2 = p["COLOR2"];
                float scale = p["SCALE"];
                tex = std::make_shared<TextureSpectrum>(
                        make_checkerboard_spectrum_texture(
                        glm::vec3(c1[0],c1[1],c1[2]),
                        glm::vec3(c2[0],c2[1],c2[2]),
                        scale, scale));
            }
            else if (type == "image") {
                std::string file = p["FILE"];
                if (file.empty()) {
                    std::cerr << "Texture \"" << name << "\": image missing FILE." << std::endl;
                    continue;
                }
                tex = std::make_shared<TextureSpectrum>(
                    make_image_spectrum_texture(name, file, texture_pool));
            }
            else {
                std::cerr << "Unsupported Texture type: " << type << std::endl;
                continue;
            }
            TexNameToID[name] = (uint32_t)textures.size();
            textures.push_back(tex);
            std::cout << "  Loaded texture: " << name << " (" << type << ")" << std::endl;
        }
    }

    std::unordered_map<std::string, uint32_t> MatNameToID;
    if (!data.contains("Materials")) {
        std::cerr << "Scene::loadFromJSON(): no Materials section." << std::endl;
        std::exit(-1);
    }
    const auto &materialsData = data["Materials"];
    for (const auto &item : materialsData.items())
    {
        const auto &name = item.key();
        const auto &p = item.value();
        Material newMaterial{};

        if (p.find("TEXTURE") != p.end()) {
            std::string tname = p["TEXTURE"];
            if (TexNameToID.find(tname) != TexNameToID.end()) {
                uint32_t tid = TexNameToID[tname];
                newMaterial.color = *textures[tid];
            } else {
                std::cerr << "Material \"" << name << "\" references unknown texture \"" << tname << "\"" << std::endl;
                const auto &col = p["RGB"];
                newMaterial.color = make_constant_spectrum_texture(
                    glm::vec3(col[0], col[1], col[2]));
            }
        }
        else {
            const auto &col = p["RGB"];
            newMaterial.color = make_constant_spectrum_texture(
                glm::vec3(col[0], col[1], col[2]));
        }

        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto &col = p["RGB"];
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto &col = p["RGB"];
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto &col = p["RGB"];
            newMaterial.hasReflective = 1.0f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Dielectric")
        {
            const auto &col = p["RGB"];
            newMaterial.hasRefractive = 1.0f;
            newMaterial.specular.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
        std::cout << "  Loaded material: " << name << " (" << p["TYPE"] << ")" << std::endl;
    }

    if (!data.contains("Objects")) {
        std::cerr << "Scene::loadFromJSON(): no Objects section." << std::endl;
        std::exit(-1);
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
                loadMeshFromOBJ(meshPath, baseGeom);
            }
            else
            {
                std::cerr << "Unsupported Mesh format: " << meshExt << ". Only .obj files are supported." << std::endl;
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
    std::cout << "  Total geometries: " << geoms.size()
        << ", emitters: " << emitters.size() << std::endl;

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

void Scene::loadFromOBJ(const std::string &objName)
{
    std::cout << "Begin OBJ Loading Mesh from: " << objName << std::endl;

    std::ifstream file(objName);
    if (!file.is_open()) {
        std::cerr << "Cannot open OBJ file: " << objName << std::endl;
        return;
    }

    // Create default material for OBJ
    Material newMaterial{};
    newMaterial.color = make_constant_spectrum_texture(glm::vec3(0.8f, 0.8f, 0.8f));
    newMaterial.emittance = 0.0f;
    newMaterial.hasReflective = 0.0f;
    newMaterial.hasRefractive = 0.0f;
    newMaterial.indexOfRefraction = 1.0f;
    uint32_t materialId = this->materials.size();
    this->materials.push_back(newMaterial);

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::string line;

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;

        if (prefix == "v") {
            // Vertex position
            float x, y, z;
            iss >> x >> y >> z;
            vertices.push_back(glm::vec3(x, y, z));
        }
        else if (prefix == "vn") {
            // Vertex normal
            float x, y, z;
            iss >> x >> y >> z;
            normals.push_back(glm::normalize(glm::vec3(x, y, z)));
        }
        else if (prefix == "vt") {
            // Texture coordinate
            float u, v;
            iss >> u >> v;
            texCoords.push_back(glm::vec2(u, v));
        }
        else if (prefix == "f") {
            // Face - parse triangle faces
            std::string vertex1, vertex2, vertex3;
            iss >> vertex1 >> vertex2 >> vertex3;

            // Parse vertex indices (format: v/vt/vn or v//vn or v/vt or v)
            auto parseVertex = [](const std::string& vertexStr) -> glm::ivec3 {
                glm::ivec3 indices(-1, -1, -1); // v, vt, vn indices
                std::istringstream viss(vertexStr);
                std::string index;
                
                int i = 0;
                while (std::getline(viss, index, '/') && i < 3) {
                    if (!index.empty()) {
                        indices[i] = std::stoi(index) - 1; // OBJ indices are 1-based
                    }
                    i++;
                }
                return indices;
            };

            glm::ivec3 v1 = parseVertex(vertex1);
            glm::ivec3 v2 = parseVertex(vertex2);
            glm::ivec3 v3 = parseVertex(vertex3);

            // Validate vertex indices
            if (v1.x < 0 || v1.x >= vertices.size() ||
                v2.x < 0 || v2.x >= vertices.size() ||
                v3.x < 0 || v3.x >= vertices.size()) {
                std::cerr << "Invalid vertex indices in OBJ file" << std::endl;
                continue;
            }

            Geom geom;
            geom.type = TRIANGLE;
            geom.materialid = materialId;

            // Set vertex positions
            geom.triangle.v0 = vertices[v1.x];
            geom.triangle.v1 = vertices[v2.x];
            geom.triangle.v2 = vertices[v3.x];

            // Set normals
            bool hasNormals = v1.z >= 0 && v2.z >= 0 && v3.z >= 0 &&
                             v1.z < normals.size() && v2.z < normals.size() && v3.z < normals.size();
            
            if (hasNormals) {
                geom.triangle.n0 = normals[v1.z];
                geom.triangle.n1 = normals[v2.z];
                geom.triangle.n2 = normals[v3.z];
            } else {
                // Calculate face normal
                glm::vec3 normal = glm::normalize(glm::cross(
                    geom.triangle.v1 - geom.triangle.v0,
                    geom.triangle.v2 - geom.triangle.v0));
                geom.triangle.n0 = normal;
                geom.triangle.n1 = normal;
                geom.triangle.n2 = normal;
            }

            // Set texture coordinates
            bool hasTexCoords = v1.y >= 0 && v2.y >= 0 && v3.y >= 0 &&
                               v1.y < texCoords.size() && v2.y < texCoords.size() && v3.y < texCoords.size();
            
            if (hasTexCoords) {
                geom.triangle.t0 = texCoords[v1.y];
                geom.triangle.t1 = texCoords[v2.y];
                geom.triangle.t2 = texCoords[v3.y];
            } else {
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
            if (materials[geom.materialid].emittance > 0.0f) {
                emitters.push_back(geom);
            }
        }
    }

    file.close();
    std::cout << "Load OBJ Mesh End: " << objName << std::endl;
    std::cout << "Loaded " << vertices.size() << " vertices, " << geoms.size() << " total triangles" << std::endl;
}

void Scene::loadMeshFromOBJ(const std::string &objPath, const Geom &baseGeom)
{
    size_t offset = geoms.size();
    loadFromOBJ(objPath);

    std::cout << "Begin Set OBJ Geoms from: " << objPath << std::endl;

    for (size_t i = offset; i < geoms.size(); i++)
    {
        geoms[i].materialid = baseGeom.materialid;

        // Transform vertices
        glm::vec4 v0 = baseGeom.transform * glm::vec4(geoms[i].triangle.v0, 1.0f);
        glm::vec4 v1 = baseGeom.transform * glm::vec4(geoms[i].triangle.v1, 1.0f);
        glm::vec4 v2 = baseGeom.transform * glm::vec4(geoms[i].triangle.v2, 1.0f);
        geoms[i].triangle.v0 = glm::vec3(v0) / v0.w;
        geoms[i].triangle.v1 = glm::vec3(v1) / v1.w;
        geoms[i].triangle.v2 = glm::vec3(v2) / v2.w;

        // Transform normals
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

    std::cout << "Set OBJ Geoms End: " << objPath << std::endl;
    std::cout << "Number of Geoms: " << geoms.size() << std::endl;
}

Scene::~Scene()
{
}
