#pragma once

#include <vector>
#include "scene.h"
#include "sceneStructs.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showImage(uchar4 *pbo, int iter);

// Getter functions for device variables needed by denoising
glm::vec3* getDevImage();
GBufferPixel* getDevGBuffer();
#if OPTIMIZED_GBUFFER
glm::mat4 getDevViewMatrix();
#endif
