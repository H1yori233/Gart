#include "main.h"
#include "preview.h"
#include <cstring>

#include "../imgui/imgui.h"
#include "../imgui/imgui_impl_glfw.h"
#include "../imgui/imgui_impl_opengl3.h"
#include "denoise.h"

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

// CHECKITOUT: simple UI parameters.
// Search for any of these across the whole project to see how these are used,
// or look at the diff for commit 1178307347e32da064dce1ef4c217ce0ca6153a8.
// For all the gory GUI details, look at commit 5feb60366e03687bfc245579523402221950c9c5.
int ui_iterations = 0;
int startupIterations = 0;
int lastLoopIterations = 0;
bool ui_showGbuffer = false;
bool ui_denoise = false;
int ui_filterSize = 80;
float ui_colorWeight = 0.45f;
float ui_normalWeight = 0.35f;
float ui_positionWeight = 0.2f;
bool ui_saveAndExit = false;
GBufferPixelType ui_gbufferType = GBUFFER_PIXEL_TYPE_TIME;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;

float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
RenderState* renderState;
int iteration;

int width;
int height;

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv)
{
    startTimeString = currentTimeString();

    if (argc < 2)
    {
        printf("Usage: %s SCENEFILE.json\n", argv[0]);
        return 1;
    }

    const char* sceneFile = argv[1];

    // Load scene file
    scene = new Scene(sceneFile);

    // Set up camera stuff from loaded path tracer settings
    iteration = 0;
    renderState = &scene->state;
    Camera& cam = renderState->camera;
    width = cam.resolution.x;
    height = cam.resolution.y;

    ui_iterations = renderState->iterations;
    startupIterations = ui_iterations;

    glm::vec3 view = cam.view;
    glm::vec3 up = cam.up;
    glm::vec3 right = glm::cross(view, up);
    up = glm::cross(right, view);

    cameraPosition = cam.position;

    // compute phi (horizontal) and theta (vertical) relative 3D axis
    // so, (0 0 1) is forward, (0 1 0) is up
    glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
    glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
    phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
    theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
    ogLookAt = cam.lookAt;
    zoom = glm::length(cam.position - ogLookAt);

    // Initialize CUDA and GL components
    init();

    // GLFW main loop
    mainLoop();

    return 0;
}

void saveImage()
{
    float samples = iteration;
    // output image file
    Image img(width, height);

    if (ui_denoise && iteration > 1) {
        cudaMemcpy(renderState->image.data(), getDevImage(),
                   width * height * sizeof(glm::vec3), cudaMemcpyDeviceToHost);
        checkCUDAError("copy denoised image for saving");
    }

    for (int x = 0; x < width; x++)
    {
        for (int y = 0; y < height; y++)
        {
            int index = x + (y * width);
            glm::vec3 pix = renderState->image[index];
            img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
        }
    }

    std::string filename = renderState->imageName;
    std::ostringstream ss;
    ss << filename << "." << startTimeString << "." << samples << "samp" << "." << INTEGRATOR_NAMES[renderState->integrator];
    filename = ss.str();

    // CHECKITOUT
    img.savePNG(filename);
    //img.saveHDR(filename);  // Save a Radiance HDR file
}

void runCuda()
{
    if (lastLoopIterations != ui_iterations) {
        lastLoopIterations = ui_iterations;
        camchanged = true;
    }

    if (camchanged)
    {
        iteration = 0;
        Camera& cam = renderState->camera;
        cameraPosition.x = zoom * sin(phi) * sin(theta);
        cameraPosition.y = zoom * cos(theta);
        cameraPosition.z = zoom * cos(phi) * sin(theta);

        cam.view = -glm::normalize(cameraPosition);
        glm::vec3 v = cam.view;
        glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
        glm::vec3 r = glm::cross(v, u);
        cam.up = glm::cross(r, v);
        cam.right = r;

        cam.position = cameraPosition;
        cameraPosition += cam.lookAt;
        cam.position = cameraPosition;
        camchanged = false;
    }

    // Map OpenGL buffer object for writing from CUDA on a single GPU
    // No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

    if (iteration == 0) {
        pathtraceFree();
        pathtraceInit(scene);
    }

    uchar4 *pbo_dptr = NULL;
    cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

    if (iteration < ui_iterations) {
        iteration++;

        // execute the kernel
        int frame = 0;
        pathtrace(frame, iteration);

        if (ui_denoise && iteration > 1) {
            applyDenoising(
                renderState->camera,
                ui_filterSize, ui_colorWeight, 
                ui_normalWeight, ui_positionWeight,
#if OPTIMIZED_GBUFFER
                getDevViewMatrix(),
#endif
                getDevImage(), getDevGBuffer()
            );
        }
    }

    if (ui_showGbuffer) {
        showGBuffer(
            pbo_dptr, ui_gbufferType, 
            renderState->camera,
#if OPTIMIZED_GBUFFER
            getDevViewMatrix(),
#endif
            getDevGBuffer()
        );
    } else {
        showImage(pbo_dptr, iteration);
    }

    // unmap buffer object
    cudaGLUnmapBufferObject(pbo);

    if (ui_saveAndExit) {
        saveImage();
        pathtraceFree();
        cudaDeviceReset();
        exit(EXIT_SUCCESS);
    }
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (action == GLFW_PRESS)
    {
        switch (key)
        {
	        case GLFW_KEY_ESCAPE:
	            saveImage();
	            glfwSetWindowShouldClose(window, GL_TRUE);
	            break;
	        case GLFW_KEY_S:
	            saveImage();
	            break;
	        case GLFW_KEY_SPACE:
	            camchanged = true;
	            renderState = &scene->state;
	            Camera& cam = renderState->camera;
	            cam.lookAt = ogLookAt;
	            break;
        }
    }
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods)
{
    if (ImGui::GetIO().WantCaptureMouse) return;

    leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
    rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
    middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (xpos == lastX || ypos == lastY)
    {
    	return; // otherwise, clicking back into window causes re-start
    }

    if (leftMousePressed)
    {
        // compute new camera parameters
        phi -= (xpos - lastX) / width;
        theta -= (ypos - lastY) / height;
        theta = std::fmax(0.001f, std::fmin(theta, PI));
        camchanged = true;
    }
    else if (rightMousePressed)
    {
        zoom += (ypos - lastY) / height;
        zoom = std::fmax(0.1f, zoom);
        camchanged = true;
    }
    else if (middleMousePressed)
    {
        renderState = &scene->state;
        Camera& cam = renderState->camera;
        glm::vec3 forward = cam.view;
        forward.y = 0.0f;
        forward = glm::normalize(forward);
        glm::vec3 right = cam.right;
        right.y = 0.0f;
        right = glm::normalize(right);

        cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
        cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
        camchanged = true;
    }

    lastX = xpos;
    lastY = ypos;
}
