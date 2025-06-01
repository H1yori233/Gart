#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// Add support for tinygltf
#define TINYGLTF_IMPLEMENTATION
#define TINYGLTF_NO_STB_IMAGE_WRITE  // We already have stb_image_write above
#define TINYGLTF_NO_STB_IMAGE        // We already have stb_image above
#include <tiny_gltf.h>
