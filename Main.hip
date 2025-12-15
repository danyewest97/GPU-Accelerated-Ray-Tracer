#include "Main.h"
#include <jni.h>

#include <hip/hip_runtime.h>

// TODO: Check if standard library functions can be used on GPU, and if not find out how to use sin and cos functions on GPU
#include <iostream> // Input and output functions like printf
#include <cstdlib> // Standard library with many useful functions
#include <chrono> // Library used for timing, for measuring performance
#include <cmath> // Standard math library for things like sine and cosine functions

// Custom/local library files
#include "main_structs.cpp" // Includes all of the required main structs and their constructors, plus some methods for them
#include "specific_structs.cpp" // Includes all of the required more-specific structs and their constructors, plus some methods for them

// #include "arrays.cpp" // Includes all of the required structs and their constructors, plus some methods for them
// NOTE: Convention will be to use custom-made array types (in arrays.cpp) for ALL arrays, and only using pointers when the pointer is pointing to a 
// SINGLE value and not multiple. This allows for bounds-checking and is much safer. --EDIT: Leaving as a suggestion for now, may implement later if it
// is necessary but I realized while thinking it over that it may end up being easier to just provide a length argument whenever we pass pointers of 
// unknown length, and implement bounds-checking using that length manually instead of making custom methods to do it


// Precondition: Matrices must be compatible for multiplication
// TODO: Confirm that matrix multiplication algorithm, specifically matrix indices, are correct, and create a standard for how matrices should be 
// initialized (should first 3 digits of 3x3 be the x-components of the 3 basis vectors, or x-, y-, and z-components of the first basis vector?)
__device__ void matrix_multiplication(double* matrix, double* vector, double* output) {
    double x = vector[0];
    double y = vector[1];
    double z = vector[2];
    double result[] = {(matrix[0] + matrix[1] + matrix[2]) * x, 
                       (matrix[3] + matrix[4] + matrix[5]) * y,
                       (matrix[6] + matrix[7] + matrix[8]) * z};
    output[0] = result[0];
    output[1] = result[1];
    output[2] = result[2];
}


// Oh yeah, baby... this is where the magic happens
__global__ void trace_ray(ray* primary_ray, triangle* triangles, color* output_color) {
    
}


// Note from self: switch to camera at (0, 0, 0) with translation (like KSA and BRUTAL) instead of translating the camera itself??
// Also need to add scaling of camera
// Same as above but takes an index instead of pixel coordinates
__device__ ray generate_camera_ray(camera curr_cam, dimensions img_dim, int index) {
    vector cam_origin = curr_cam.origin;
    vector cam_normal = curr_cam.rotation;
    int width = img_dim.width;
    int height = img_dim.height;

    // Converting the index into pixel coordinates
    int pixel_x = index % width;
    int pixel_y = (int) ((double) index / width);

    // TODO: Confirm that rays need to be offset? Or should they just come directly from the camera origin with no offset? I think having an offset is 
    // correct
    // Note: may need optimizations for greater performance -- and either offload to CPU or do this in parallel (make new kernel for this?)
    double passthrough_plane_distance = curr_cam.fov_scale;              // Corresponds to the FOV scale of the camera (vertical and horizontal FOV 
    // values depend on the width and height of the camera), this is my own 
    // method for generating camera rays: imagine a plane that is fov_scale 
    // units from the camera pinhole/aperture, and we are drawing a ray to the 
    // centers of each of the cells on this plane -- the further away this 
    // plane gets, the more clustered the rays are, so the smaller the FOV, 
    // and vice versa for when the plane gets closer to the camera.
    // I chose to do it this way because it is very intuitive to me and 
    // incredibly easy to implement, since no rotation is involved
    
    double x = (-(double) (width) / 2) + pixel_x + 0.5;
    double y = (-(double) (height) / 2) + pixel_y + 0.5;
    vector true_origin(0, 0, 0);                              // We will initially draw rays originating from (0, 0, 0), then translate 
    // and rotate them according to the camera's position
    vector ray_direction(x, y, passthrough_plane_distance);      // Adding a 0.5 unit offset to position rays into the middle of each
                                                                                // pixel, instead of the top-right corner without the offset
    
    
    ray curr_ray(true_origin, ray_direction);                    // Creating the initial ray, not yet conformed to the camera's orientation
    
    cam_origin.add(curr_ray.origin);                                      // Orienting the ray to be lined up properly with the camera (translating 
    // to the camera's origin)
    
    // TODO: Make this into one operation, by precalculating the single matrix needed for all three rotations and only using that matrix
    curr_ray.direction.rotate_x(true_origin, cam_normal.x);              // Rotating the ray to face the same direction as the camera -- using 
                                                                            // true_origin instead of cam_origin because the direction of the ray 
                                                                            // determines where it points from its origin, so it is not 
                                                                            // location-dependent and should therefore be rotated about (0, 0, 0)
    curr_ray.direction.rotate_y(true_origin, cam_normal.y);
    curr_ray.direction.rotate_z(true_origin, cam_normal.z);
    curr_ray.direction.normalize();                                         // Normalizing the ray's direction so that distances returned from intersection methods will be 
                                                                            // absolute and not scaled by the ray direction's length (the t-value, or distance, returned from the 
                                                                            // ray-plane and ray-triangle intersection methods is dependent upon the ray direction's length, so if 
                                                                            // that length is 1, the t-value returned will be the same as the Euclidean distance from the ray's 
                                                                            // origin to the intersection point)

    return curr_ray;
}


// Note: For some reason (probably a compilation bug or something), HIP seems to break when I put two identical print statements in here
// -- so don't do that!
__global__ void test_kernel(color* img_out, camera cam, dimensions img_dimensions, triangle* triangles, box_node* root_node, int num_tris, int num_pixels, int num_threads) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    // printf("test kernel started\n");

    // printf("%f\n", root_node->children[0].box.min_x);

    while (global_index < num_pixels) {
        // Creating coalesced memory reads (where thread 1 reads index 1, thread 2 index 2, thread 1 index 3, thread 2 index 4, etc) by starting 
        // with the global index of the thread and adding num_threads each time we calculate a pixel, to offset memory reads so that they are 
        // coalesced and faster (see here: https://stackoverflow.com/questions/5041328/in-cuda-what-is-memory-coalescing-and-how-is-it-achieved)
        // Essentially this means that memory accesses only occur on addresses adjacent to previous memory accesses, to increase memory speed
        
        int i = global_index;
        global_index += num_threads;


        ray primary_ray = generate_camera_ray(cam, img_dimensions, i);
        bool has_an_intersection = false;
        for (int j = 0; j < num_tris; j++) {
            hit collision = ray_triangle_intersection(primary_ray, root_node);
            has_an_intersection |= collision.has_intersection;
        }

        __syncthreads();
        if (has_an_intersection) {
            color white(1, 1, 1);
            img_out[i] = white;                                 // VERY slow, unsure why or how to solve but needs to be optimized
        }
    }
    

    // printf("test kernel finished\n");
}


__global__ void initialize() {};

// Important note: not sure how much overhead variable creation and logic happening inside the kernel is adding, because I am not just sending
// a shader to the GPU for it to handle and send back, but actually making new variables and doing more than *just* matrix matrix multiplication
// on the kernel

color* run(int width, int height)
{
    // initialize<<<1, 1>>>();

    // Note: I have discovered that structs can, in fact, be passed by value to the GPU even if they contain other structs as members. 
    // Would've saved me a WHOLE lot of time if I had realized this earlier

    // The number of pixels and size of our output image
    int num_pixels = width * height;
    int img_size = num_pixels * sizeof(color);              // The amount of memory we need for our image
    color* img_out;                                         // The output image, we are not initializing it with any memory because we want it to be
                                                             // initialized in the device/GPU's memory, not the host/CPU's memory
    hipMalloc(&img_out, img_size);

    // Assigning all of our variables -- things like camera settings and test triangles
    double fov_scale = 1;
    vector cam_origin(0, 0, 0);
    vector cam_direction(0, 0, 0);
    camera main_cam(cam_origin, cam_direction, fov_scale);

    dimensions img_dim(width, height);

    int num_tris = 1;
    int size_of_triangles = num_tris * sizeof(triangle);
    triangle* triangles = new triangle[num_tris];
    
    srand(time(0));                                             // Generating a pseudo-random seed depending on the time each iteration
    material placeholder_material(color(255, 255, 255), 1, 0, 0);
    for (int i = 0; i < num_tris; i++) {
        vector a((rand() % 101) - 50, (rand() % 101) - 50, 1);
        vector b(a.x + (rand() % 31) - 15, a.y + (rand() % 31) - 15, 1);
        vector c((rand() % 51) - 25, (rand() % 51) - 25, 1);

        triangle test_tri_1(placeholder_material, a, b, c);
        triangles[i] = test_tri_1;
    }

    // Copying our triangle data to the GPU
    triangle* gpu_triangles;
    hipMalloc(&gpu_triangles, size_of_triangles);
    hipMemcpy(gpu_triangles, triangles, size_of_triangles, hipMemcpyHostToDevice);
    

    box_node* root_node = generate_bounding_box_tree(triangles, num_tris);
    box_node* gpu_node = prepare_box_node_for_gpu(root_node);


    auto start = std::chrono::high_resolution_clock::now();

    int num_threads = 400;

    // Starting the kernel
    test_kernel<<<num_threads, 1>>>(img_out, main_cam, img_dim, gpu_triangles, gpu_node, num_tris, num_pixels, num_threads);               // Start index is inclusive, end index is exclusive
    
    // Wait on all active streams on the current device. VERY NECESSARY
    hipDeviceSynchronize();
    
    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "\n";
    printf("frame");
    
    // Getting our final result from the GPU to the CPU
    color* result = new color[img_size];
    hipMemcpy(result, img_out, img_size, hipMemcpyDeviceToHost);

    hipDeviceReset();




    return result;
}




JNIEXPORT jdoubleArray JNICALL Java_Main_test(JNIEnv* env, jobject thisObject, jint width, jint height) {
    color* img = run(width, height);

    int num_pixels = width * height;
    int num_colors = num_pixels * 3;
    jdoubleArray img_java = env->NewDoubleArray(num_colors);

    // Creating a usable shallow copy of the img_java array as a jdouble pointer, and filling it with the color values calculated by the GPU to send 
    // back to the Java host program to be displayed
    jboolean is_copy[1] = {JNI_FALSE};
    jdouble* arr_ptrs = env->GetDoubleArrayElements(img_java, is_copy);              // Getting the img_out jdoubleArray object as a C++ array
    for (int i = 0; i < num_pixels; i++) {
        // Taking the color values from each pixel
        color curr_color = img[i];
        double r = curr_color.r;
        double g = curr_color.g;
        double b = curr_color.b;

        int arr_idx = i * 3;

        // Copying the color values to the arr_ptrs array (jdouble == double, basically, so we don't need to convert manually at all)
        arr_ptrs[arr_idx] = r;
        arr_ptrs[arr_idx + 1] = g;
        arr_ptrs[arr_idx + 2] = b;
    }

    int copy_changes_to_array_mode_number = 0;                                          // Signifies the mode for the array release, in this case 
                                                                                        // setting the mode to 0, which corresponds to copying back 
                                                                                        // all of the changes we made in arr_ptrs to img_java, and 
                                                                                        // then freeing arr_ptrs, when Release...() is called below
    env->ReleaseDoubleArrayElements(img_java, arr_ptrs, copy_changes_to_array_mode_number);     // Copying back arr_ptrs to img_java to update changes
    return img_java;
}
