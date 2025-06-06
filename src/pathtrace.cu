#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/device_ptr.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include "denoise.h"

/**
 * MATERIAL-BASED SORTING OPTIMIZATION:
 * 
 * When SORT_BY_MATERIAL is enabled (set to 1), path segments and intersections
 * are sorted by material ID before shading. This optimization addresses the 
 * performance issue of warp divergence in the shading kernel.
 * 
 * PROBLEM:
 * - In a typical scene, adjacent pixels may hit different materials
 * - When threads in the same warp execute different material/BSDF code paths,
 *   they diverge and execute serially, reducing parallel efficiency
 * 
 * SOLUTION:
 * - Sort rays by material ID so threads with the same material are contiguous
 * - Threads in the same warp will execute the same BSDF code path
 * - This reduces warp divergence and improves parallel efficiency
 * 
 * CRITICAL IMPLEMENTATION DETAIL:
 * - PathSegment and ShadeableIntersection arrays MUST be sorted together
 * - Using zip_iterator ensures paths[i] ↔ intersections[i] correspondence is maintained
 * - Previous buggy approach: sorting arrays separately broke this correspondence
 * 
 * PERFORMANCE IMPACT:
 * - Adds sorting overhead each bounce (~O(n log n) per bounce)
 * - Reduces warp divergence in shading kernel (can be significant speedup)
 * - Net performance depends on material diversity and scene complexity
 * - Most beneficial in scenes with many different materials
 * 
 * TOGGLE: Set SORT_BY_MATERIAL to 0 to disable, 1 to enable
 */

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index];

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static GBufferPixel* dev_gBuffer = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...

#if OPTIMIZED_GBUFFER
static glm::mat4 dev_viewMatrix = glm::mat4(1.0f);
#endif

#if SORT_BY_MATERIAL
static int* dev_material_keys = NULL;
#endif

static LightInfo* dev_lightInfos = NULL;

// ------------------------------------------------------------

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    cudaMalloc(&dev_gBuffer, pixelcount * sizeof(GBufferPixel));

    // TODO: initialize any extra device memeory you need

    denoiseInit(pixelcount);
#if SORT_BY_MATERIAL
    cudaMalloc(&dev_material_keys, pixelcount * sizeof(int));
#endif
    int num_lights = scene->lightInfos.size();
    cudaMalloc(&dev_lightInfos, num_lights * sizeof(LightInfo));
    cudaMemcpy(dev_lightInfos, scene->lightInfos.data(), num_lights * sizeof(LightInfo), cudaMemcpyHostToDevice);

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    cudaFree(dev_gBuffer);
    // TODO: clean up any extra device memory you created

    denoiseFree();
#if SORT_BY_MATERIAL
    cudaFree(dev_material_keys);
#endif
    cudaFree(dev_lightInfos);

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
        thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> uPixel(-0.5, 0.5);
        float jitter_x = uPixel(rng);
        float jitter_y = uPixel(rng);
        segment.ray.direction = glm::normalize(cam.view
            - cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f + jitter_x)
            - cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f + jitter_y)
        );

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;
        bool outside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == QUAD)
            {
                t = quadIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == TRIANGLE)
            {
                t = triangleIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
            }
        }

        if (hit_geom_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            intersections[path_index].surfaceNormal = normal;
        }
    }
}

// ------------------------------------------------------------

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
                scatterRay(pathSegments[idx], intersect, intersection.surfaceNormal, material, rng);

#if RUSSIAN_ROULETTE
                if(pathSegments[idx].remainingBounces > 0 && depth > RR_DEPTH) {
                    float throughput = luminance(pathSegments[idx].color);
                    float prob = fminf(0.95f, fmaxf(0.05f, throughput));
        
                    if(u01(rng) < prob) {
                        pathSegments[idx].color /= prob;
                        pathSegments[idx].remainingBounces --;
                    } else {
                        pathSegments[idx].remainingBounces = 0;
                    }
                }
#else
                pathSegments[idx].remainingBounces--;
#endif
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

__global__ void shadeNEE(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials,
    Geom* geoms,
    int geoms_size,
    LightInfo* lightInfos,
    int num_lights,
    float total_light_power,
    int depth)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
                pathSegments[idx].remainingBounces = 0;
                return;
            }

            // Calculate intersection point and normal
            glm::vec3 intersect = getPointOnRay(pathSegments[idx].ray, intersection.t);
            glm::vec3 normal = intersection.surfaceNormal;
            glm::vec3 wi = -pathSegments[idx].ray.direction; // incident direction

            glm::vec3 totalRadiance = glm::vec3(0.0f);
            
            // Light Sampling
            float selectionPdf;
            int lightIdx = sampleLight(lightInfos, num_lights, total_light_power, 
                                       rng, selectionPdf);
            if (lightIdx >= 0 && selectionPdf > 0.0f) {
                // Sample a point on the light
                int geomId = lightInfos[lightIdx].geomid;
                Geom& lightGeom = geoms[geomId];
                glm::vec3 lightNormal;
                float areaPdf;
                glm::vec3 lightPoint = samplePointOnGeom(lightGeom, intersect, rng, lightNormal, areaPdf);

                // Calculate direction and distance to light
                glm::vec3 lightDir = lightPoint - intersect;
                float lightDistance = glm::length(lightDir);
                lightDir /= lightDistance; // normalize

                // Shadow test
                Ray shadowRay;
                shadowRay.origin = intersect + 0.001f * normal; // offset to prevent self-intersection
                shadowRay.direction = lightDir;
                
                bool inShadow = shadowTest(shadowRay, geoms, geoms_size, lightDistance);
                if (!inShadow) {
                    // Calculate geometry terms
                    float cosTheta_light = fmaxf(0.0f, -glm::dot(lightDir, lightNormal));
                    float cosTheta_surface = fmaxf(0.0f, glm::dot(lightDir, normal));
                    
                    if (cosTheta_light > 0.0f && cosTheta_surface > 0.0f) {
                        float geometryTerm = cosTheta_light * cosTheta_surface / (lightDistance * lightDistance);
                        
                        // Evaluate BSDF
                        glm::vec3 bsdfValue = evaluateBSDF(material, wi, lightDir, normal);
                        
                        // Light emission
                        glm::vec3 emission = lightInfos[lightIdx].emission;
                        
                        // Calculate MIS weights
                        float p1 = selectionPdf * areaPdf; // light sampling pdf
                        float p2 = pdfBSDF(material, wi, lightDir, normal) * geometryTerm / cosTheta_surface; // BSDF sampling pdf converted to area measure
                        
                        float w1 = (p1 * p1) / (p1 * p1 + p2 * p2); // MIS weight
                        
                        if (p1 > 0.0f) {
                            glm::vec3 directLight = w1 * geometryTerm * bsdfValue * emission / p1;
                            totalRadiance += directLight;
                        }
                    }
                }
            }

            // BSDF Sampling
            glm::vec3 wo; // outgoing direction
            float bsdfPdf;
            glm::vec3 bsdfValue = sampleBSDF(material, wi, normal, rng, wo, bsdfPdf);
            
            if (bsdfPdf > 0.0f) {
                // Update ray
                pathSegments[idx].ray.origin = intersect + 0.001f * normal;
                pathSegments[idx].ray.direction = wo;
                
                // Calculate throughput
                if(material.hasReflective > 0.0f) {
                    pathSegments[idx].color *= bsdfValue;
                } else {
                    float cosTheta = fmaxf(0.0f, glm::dot(wo, normal));
                    pathSegments[idx].color *= bsdfValue * cosTheta / bsdfPdf;

                    // Add direct lighting contribution
                    pathSegments[idx].color += totalRadiance;
                }
            } else {
                pathSegments[idx].remainingBounces = 0;
            }

            // Russian Roulette termination
#if RUSSIAN_ROULETTE
            if(pathSegments[idx].remainingBounces > 0 && depth > RR_DEPTH) {
                float throughput = luminance(pathSegments[idx].color);
                float prob = fminf(0.95f, fmaxf(0.05f, throughput));
    
                if(u01(rng) < prob) {
                    pathSegments[idx].color /= prob;
                    pathSegments[idx].remainingBounces--;
                } else {
                    pathSegments[idx].remainingBounces = 0;
                }
            } else {
                pathSegments[idx].remainingBounces--;
            }
#else
            pathSegments[idx].remainingBounces--;
#endif
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
            pathSegments[idx].remainingBounces = 0;
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        image[iterationPath.pixelIndex] += iterationPath.color;
    }
}

struct pathExists {
    __host__ __device__ bool operator() (const PathSegment& pathSegment){
        return pathSegment.remainingBounces > 0;
    }
};

#if SORT_BY_MATERIAL
// Functor to extract material ID from ShadeableIntersection for sorting
struct MaterialIdExtractor {
    __host__ __device__ int operator()(const ShadeableIntersection& intersection) const {
        // Use material ID as sort key, with a special case for missed rays (t < 0)
        return (intersection.t > 0.0f) ? intersection.materialId : -1;
    }
};

// Kernel to extract material IDs for sorting
__global__ void extractMaterialIds(int num_paths, ShadeableIntersection* intersections, int* material_keys) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        // Use material ID as sort key, with -1 for missed rays
        material_keys[idx] = (intersections[idx].t > 0.0f) ? intersections[idx].materialId : -1;
    }
}
#endif

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(int frame, int iter) 
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

#if OPTIMIZED_GBUFFER
    dev_viewMatrix = glm::lookAt(cam.position, cam.position + cam.view, cam.up);
#endif

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    // Empty gbuffer
    cudaMemset(dev_gBuffer, 0, pixelcount * sizeof(GBufferPixel));

	// clean shading chunks
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections<<<numblocksPathSegmentTracing, blockSize1d>>> (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();

        if (depth == 0) {
            generateGBuffer(
                num_paths, 
                dev_intersections, 
                dev_paths, 
#if OPTIMIZED_GBUFFER
                dev_viewMatrix,
#endif
                dev_gBuffer
            );
        }
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if SORT_BY_MATERIAL
        // MATERIAL SORTING OPTIMIZATION:
        // Sort path segments and intersections by material ID to reduce warp divergence.
        // CRITICAL: Both arrays must be sorted together to maintain paths[i] & intersections[i] correspondence
        
        // Extract material IDs to use as sort keys
        dim3 numBlocksExtract = (num_paths + blockSize1d - 1) / blockSize1d;
        extractMaterialIds<<<numBlocksExtract, blockSize1d>>>(num_paths, dev_intersections, dev_material_keys);
        checkCUDAError("extract material IDs");
        
        // Create thrust device pointers
        thrust::device_ptr<int> keys_ptr(dev_material_keys);
        thrust::device_ptr<PathSegment> paths_ptr(dev_paths);
        thrust::device_ptr<ShadeableIntersection> intersections_ptr(dev_intersections);
        
        // Use zip_iterator to sort both arrays together with the same permutation
        auto zipped_begin = thrust::make_zip_iterator(thrust::make_tuple(paths_ptr, intersections_ptr));
        thrust::sort_by_key(thrust::device, keys_ptr, keys_ptr + num_paths, zipped_begin);
        
        // Both arrays are now sorted by material ID while maintaining correspondence
        PathSegment* paths_to_shade = dev_paths;
        ShadeableIntersection* intersections_to_shade = dev_intersections;
#else
        // Use original arrays for shading (no sorting)
        PathSegment* paths_to_shade = dev_paths;
        ShadeableIntersection* intersections_to_shade = dev_intersections;
#endif

        switch (hst_scene->state.integrator) {
            case INTEGRATOR_FAKE:
                shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
                    iter,
                    num_paths,
                    intersections_to_shade,
                    paths_to_shade,
                    dev_materials,
                    depth
                );
                break;
            
            case INTEGRATOR_NAIVE:
                shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
                    iter,
                    num_paths,
                    intersections_to_shade,
                    paths_to_shade,
                    dev_materials,
                    depth
                );
                break;
                
            case INTEGRATOR_NEE:
            default:
                shadeNEE<<<numblocksPathSegmentTracing, blockSize1d>>>(
                    iter,
                    num_paths,
                    intersections_to_shade,
                    paths_to_shade,
                    dev_materials,
                    dev_geoms,
                    hst_scene->geoms.size(),
                    dev_lightInfos,
                    hst_scene->lightInfos.size(),
                    hst_scene->totalLightPower,
                    depth
                );
                break;
        }
        checkCUDAError("shade material");

        dev_path_end = thrust::partition(thrust::device, dev_paths, dev_path_end, pathExists());
        num_paths = dev_path_end - dev_paths;
        // iterationComplete = true; // TODO: should be based off stream compaction results.
        if (num_paths <= 0 || depth >= traceDepth) {
            iterationComplete = true;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    // finalGather<<<numBlocksPixels, blockSize1d>>>(num_paths, dev_image, dev_paths);
    /*
        num_paths has been modified after compaction `num_paths = dev_path_end - dev_paths;`
        so we can not use it any more, as `int num_paths = dev_path_end - dev_paths;`
        we can use pixelcount instead
    */
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}

void showImage(uchar4* pbo, int iter) {
    const Camera &cam = hst_scene->state.camera;
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
            (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
            (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);
}

// Getter functions for device variables needed by denoising
glm::vec3* getDevImage() {
    return dev_image;
}

GBufferPixel* getDevGBuffer() {
    return dev_gBuffer;
}

#if OPTIMIZED_GBUFFER
glm::mat4 getDevViewMatrix() {
    return dev_viewMatrix;
}
#endif
