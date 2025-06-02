#pragma once

#include <vector>
#include "scene.h"

void pathtraceInit(Scene *scene);
void pathtraceFree();
void pathtrace(int frame, int iteration);
void showGBuffer(uchar4 *pbo, GBufferPixelType type);
void showImage(uchar4 *pbo, int iter);
void applyDenoising(int width, int height, int filterSize, float colorWeight, float normalWeight, float positionWeight);
