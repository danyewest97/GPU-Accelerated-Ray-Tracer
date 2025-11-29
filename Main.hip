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
#include "gpu_copying.cpp" // Includes all of the required functions for copying structs AND THEIR MEMBERS*** over to the GPU

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


// __global__ 


// Oh yeah, baby... this is where the magic happens
__global__ void trace_ray(ray* primary_ray, triangle* triangles, color* output_color) {
    
}

// Deprecated
// Takes the given camera and image dimensions and generates the corresponding primary/camera rays for them
__device__ ray** generate_camera_rays(camera* curr_cam, dimensions* img_dims) {
    vector* cam_origin = curr_cam->origin;
    vector* cam_normal = curr_cam->rotation;
    double* fov_scale = curr_cam->fov_scale;
    
    int* width = img_dims->width;
    int* height = img_dims->height;
    int num_rays = *width * *height;
    

    // TODO: Confirm that rays need to be offset? Or should they just come directly from the camera origin with no offset? I think having an offset is 
    // correct
    // Note: may need optimizations for greater performance -- and either offload to CPU or do this in parallel (make new kernel for this?)
    double initial_horizontal_offset = -*width / 2 + 0.5;                    // Adding 0.5 to each offset to move the rays to be in the middle of each 
                                                                            // pixel
    double initial_vertical_offset = -*height / 2 + 0.5;
    double* passthrough_plane_distance = fov_scale;                          // Corresponds to the FOV scale of the camera (vertical and horizontal FOV 
                                                                            // values depend on the width and height of the camera), this is my own 
                                                                            // method for generating camera rays: imagine a plane that is fov_scale 
                                                                            // units from the camera pinhole/aperture, and we are drawing a ray to the 
                                                                            // centers of each of the cells on this plane -- the further away this 
                                                                            // plane gets, the more clustered the rays are, so the smaller the FOV, 
                                                                            // and vice versa for when the plane gets closer to the camera.
                                                                            // I chose to do it this way because it is very intuitive to me and incredibly easy to implement, since no rotation is involved

    vector* start_position = new vector(initial_horizontal_offset, 
                                        initial_vertical_offset, 
                                        *passthrough_plane_distance);        // The position on the plane (described above) that we start on, drawing
                                                                            // a ray from the camera to this position will generate our camera rays
    vector* true_origin = new vector(0, 0, 0);                              // We will initially draw rays originating from (0, 0, 0), then translate 
                                                                            // and rotate them according to the camera's position
    ray** primary_rays = new ray*[num_rays];                                // Also called camera rays, primary rays are the resulting, final rays 
                                                                            // that we will use for ray casting
    
    // Iterating through every cell in the image and creating a camera/primary ray for it
    for (int i = 0; i < *width; i++) {
        for (int j = 0; j < *height; j++) {
            vector* curr_position = start_position->clone();
            ray* curr_ray = new ray(true_origin, curr_position);            // Creating the initial ray, not yet conformed to the camera's orientation

            cam_origin->add(curr_ray->origin);                      // Orienting the ray to be lined up properly with the camera (translating 
                                                                    // to the camera's origin)
            // TODO: Make this into one operation, by precalculating the single matrix needed for all three rotations and only using that matrix
            curr_ray->direction->rotate_x(true_origin, *cam_normal->x);     // Rotating the ray to face the same direction as the camera -- using 
                                                                            // true_origin instead of cam_origin because the direction of the ray 
                                                                            // determines where it points from its origin, so it is not 
                                                                            // location-dependent and should therefore be rotated about (0, 0, 0)
            curr_ray->direction->rotate_y(true_origin, *cam_normal->y);
            curr_ray->direction->rotate_z(true_origin, *cam_normal->z);
            curr_ray->direction->normalize();                                 // Normalizing the ray's direction so that distances returned from intersection methods will be 
                                                                            // absolute and not scaled by the ray direction's length (the t-value, or distance, returned from the 
                                                                            // ray-plane and ray-triangle intersection methods is dependent upon the ray direction's length, so if 
                                                                            // that length is 1, the t-value returned will be the same as the Euclidean distance from the ray's 
                                                                            // origin to the intersection point)
            primary_rays[i * *height + j] = curr_ray;
            *start_position->y++;
        }
        *start_position->x++;
    }

    return primary_rays;
}


// Same as above but only calculates a single ray, for parallel processing
__device__ ray* generate_camera_ray(camera* curr_cam, dimensions* img_dim, int pixel_x, int pixel_y) {
    vector* cam_origin = curr_cam->origin;
    vector* cam_normal = curr_cam->rotation;
    int* width = img_dim->width;
    int* height = img_dim->height;

    // TODO: Confirm that rays need to be offset? Or should they just come directly from the camera origin with no offset? I think having an offset is 
    // correct
    // Note: may need optimizations for greater performance -- and either offload to CPU or do this in parallel (make new kernel for this?)
    double* passthrough_plane_distance = curr_cam->fov_scale;              // Corresponds to the FOV scale of the camera (vertical and horizontal FOV 
    // values depend on the width and height of the camera), this is my own 
    // method for generating camera rays: imagine a plane that is fov_scale 
    // units from the camera pinhole/aperture, and we are drawing a ray to the 
    // centers of each of the cells on this plane -- the further away this 
    // plane gets, the more clustered the rays are, so the smaller the FOV, 
    // and vice versa for when the plane gets closer to the camera.
    // I chose to do it this way because it is very intuitive to me and 
    // incredibly easy to implement, since no rotation is involved
    
    double x = (-(double) (*width) / 2) + pixel_x + 0.5;
    double y = (-(double) (*height) / 2) + pixel_y + 0.5;
    vector* true_origin = new vector(0, 0, 0);                              // We will initially draw rays originating from (0, 0, 0), then translate 
    // and rotate them according to the camera's position
    vector* ray_direction = new vector(x, y, *passthrough_plane_distance);      // Adding a 0.5 unit offset to position rays into the middle of each
                                                                                // pixel, instead of the top-right corner without the offset
    
    
    ray* curr_ray = new ray(true_origin, ray_direction);                    // Creating the initial ray, not yet conformed to the camera's orientation
    
    cam_origin->add(curr_ray->origin);                                      // Orienting the ray to be lined up properly with the camera (translating 
    // to the camera's origin)
    
    // TODO: Make this into one operation, by precalculating the single matrix needed for all three rotations and only using that matrix
    //curr_ray->direction->rotate_x(true_origin, *cam_normal->x);              // Rotating the ray to face the same direction as the camera -- using 
                                                                            // true_origin instead of cam_origin because the direction of the ray 
                                                                            // determines where it points from its origin, so it is not 
                                                                            // location-dependent and should therefore be rotated about (0, 0, 0)
    //curr_ray->direction->rotate_y(true_origin, *cam_normal->y);
    //curr_ray->direction->rotate_z(true_origin, *cam_normal->z);
    curr_ray->direction->normalize();                                         // Normalizing the ray's direction so that distances returned from intersection methods will be 
                                                                            // absolute and not scaled by the ray direction's length (the t-value, or distance, returned from the 
                                                                            // ray-plane and ray-triangle intersection methods is dependent upon the ray direction's length, so if 
                                                                            // that length is 1, the t-value returned will be the same as the Euclidean distance from the ray's 
                                                                            // origin to the intersection point)

    return curr_ray;
}


// Note from self: switch to camera at (0, 0, 0) with translation (like KSA and BRUTAL) instead of translating the camera itself??
// Also need to add scaling of camera
// Same as above but takes an index instead of pixel coordinates
__device__ ray* generate_camera_ray(camera* curr_cam, dimensions* img_dim, int index) {
    vector* cam_origin = curr_cam->origin;
    vector* cam_normal = curr_cam->rotation;
    int* width = img_dim->width;
    int* height = img_dim->height;

    // Converting the index into pixel coordinates
    int pixel_x = index % *width;
    int pixel_y = (int) ((double) index / *width);

    // TODO: Confirm that rays need to be offset? Or should they just come directly from the camera origin with no offset? I think having an offset is 
    // correct
    // Note: may need optimizations for greater performance -- and either offload to CPU or do this in parallel (make new kernel for this?)
    double* passthrough_plane_distance = curr_cam->fov_scale;              // Corresponds to the FOV scale of the camera (vertical and horizontal FOV 
    // values depend on the width and height of the camera), this is my own 
    // method for generating camera rays: imagine a plane that is fov_scale 
    // units from the camera pinhole/aperture, and we are drawing a ray to the 
    // centers of each of the cells on this plane -- the further away this 
    // plane gets, the more clustered the rays are, so the smaller the FOV, 
    // and vice versa for when the plane gets closer to the camera.
    // I chose to do it this way because it is very intuitive to me and 
    // incredibly easy to implement, since no rotation is involved
    
    double x = (-(double) (*width) / 2) + pixel_x + 0.5;
    double y = (-(double) (*height) / 2) + pixel_y + 0.5;
    vector* true_origin = new vector(0, 0, 0);                              // We will initially draw rays originating from (0, 0, 0), then translate 
    // and rotate them according to the camera's position
    vector* ray_direction = new vector(x, y, *passthrough_plane_distance);      // Adding a 0.5 unit offset to position rays into the middle of each
                                                                                // pixel, instead of the top-right corner without the offset
    
    
    ray* curr_ray = new ray(true_origin, ray_direction);                    // Creating the initial ray, not yet conformed to the camera's orientation
    
    cam_origin->add(curr_ray->origin);                                      // Orienting the ray to be lined up properly with the camera (translating 
    // to the camera's origin)
    
    // TODO: Make this into one operation, by precalculating the single matrix needed for all three rotations and only using that matrix
    //curr_ray->direction->rotate_x(true_origin, *cam_normal->x);              // Rotating the ray to face the same direction as the camera -- using 
                                                                            // true_origin instead of cam_origin because the direction of the ray 
                                                                            // determines where it points from its origin, so it is not 
                                                                            // location-dependent and should therefore be rotated about (0, 0, 0)
    //curr_ray->direction->rotate_y(true_origin, *cam_normal->y);
    //curr_ray->direction->rotate_z(true_origin, *cam_normal->z);
    curr_ray->direction->normalize();                                         // Normalizing the ray's direction so that distances returned from intersection methods will be 
                                                                            // absolute and not scaled by the ray direction's length (the t-value, or distance, returned from the 
                                                                            // ray-plane and ray-triangle intersection methods is dependent upon the ray direction's length, so if 
                                                                            // that length is 1, the t-value returned will be the same as the Euclidean distance from the ray's 
                                                                            // origin to the intersection point)

    return curr_ray;
}


__global__ void ray_trace_kernel(double* vertices_in, double* img_out, double size, double triangles_per_thread, double img_width, double img_height) {
    int global_index = threadIdx.x + blockIdx.x * blockDim.x;
    
}


// Note: For some reason (probably a compilation bug or something), HIP seems to break when I put two identical print statements in here
// -- so don't do that!
__global__ void test_kernel(color** img_out, camera* cam, dimensions* img_dimensions, triangle** triangles, int num_tris, int start_index, int end_index) {
    printf("test kernel started\n");


    for (int i = start_index; i < end_index; i++) {
        ray* primary_ray = generate_camera_ray(cam, img_dimensions, i);
        bool* has_an_intersection = new bool[1];
        for (int j = 0; j < num_tris; j++) {
            triangle* curr_tri = triangles[j];
            bool* has_intersection = new bool[1];
            double* t_out = new double[1];
            vector* collision_point = ray_triangle_intersection_t(primary_ray, curr_tri, has_intersection, t_out);
            *has_an_intersection |= *has_intersection;
        }

        if (*has_an_intersection) {
            // IDK why this way doesn't work
            //color* white = new color(1, 1, 1);
            //img_out[i] = white;

            // NOTE: IDK why I have to do it this way (why just doing img_out = new color(1, 1, 1) doesn't work), I have some suspicions but it feels
            // wrong to have to do it this way. There must be a better way.
            *(img_out[i]->r) = 1;
            *(img_out[i]->g) = 1;
            *(img_out[i]->b) = 1;
        }
    }
    

    

    printf("test kernel finished\n");
}


__global__ void initialization() {}


// Runs the test kernel iterations times and spits out the average time per iteration, in nanoseconds
// Needs to be updated
void run_test_kernel(int iterations) {
    int* times = new int[iterations];                                                   // An array to store the results of the test_kernel running 
                                                                                        // multiple times, to take the average time
    for (int i = iterations; i >= 0; i--) {
        auto start = std::chrono::high_resolution_clock::now();                         // For timing, to see how long the test function takes to run
        
        //test_kernel<<<
        //    dim3(1),
        //    dim3(1),
        //    0,
        //    hipStreamDefault
        //>>>(new color[1]);
        
        // Wait on all active streams on the current device. VERY NECESSARY
        hipDeviceSynchronize();
        auto finish = std::chrono::high_resolution_clock::now();
        //std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() << "ns\n";
        if (i < iterations) {           // Throwing out the first time because it still seems to have some baseline delay before running ¯\_(ツ)_/¯
            times[i] = std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count();
        }
    }
    
    int sum = 0;
    for (int i = 0; i < iterations; i++) {
        sum += times[i];
    }
    printf("%i\n", sum / iterations);
}


// Important note: not sure how much overhead variable creation and logic happening inside the kernel is adding, because I am not just sending
// a shader to the GPU for it to handle and send back, but actually making new variables and doing more than *just* matrix matrix multiplication
// on the kernel

color** run(int width, int height)
{
    // Kind of a hack, but see note below -- HIP takes a very long time to run the first kernel, but not the ones run after it, so I am including this
    // call to an empty kernel to "initialize" HIP so that the timing for the test_kernel kernel is not offset for debugging/timing purposes
    initialization<<<
        dim3(1),
        dim3(1),
        0,
        hipStreamDefault
    >>>();
    hipDeviceSynchronize();

    // Note: For some reason, it seems that whenever dynamically allocating memory on the GPU (i.e. with "new"), it adds a baseline ~13 milliseconds
    // to the runtime -- NOT 13 milliseconds per allocation, but 13 milliseconds no matter the amount of allocations happening, if there is at least
    // one allocation.
    // In other words, if you don't allocate anything dynamically, the runtime stays low, but as soon as you make even a bool* with a single bit of 
    // memory using the "new" operator, runtime jumps up by 13 ms, and it stays that way no matter how much more memory you allocate.
    // ALSO: When running a kernel multiple times in a for-loop, the base runtime is around 500 (!!) ms, even running an empty kernel. However, I 
    // think this is just an "initialization" period that HIP needs to start communicating with the GPU, or something similar, because it isn't there 
    // for further runs of the same kernel -- only the first run.
    
    //run_test_kernel(1);
    
    // The number of pixels and size of our output image
    int num_pixels = width * height;
    int img_size = num_pixels * sizeof(color*);              // The amount of memory we need for our image
    color** img_out;                                         // The output image, we are not initializing it with any memory because we want it to be
                                                             // initialized in the device/GPU's memory, not the host/CPU's memory
    hipMalloc(&img_out, img_size);

    // Filling the img with black pixels
    color** blank_image = new color*[num_pixels];
    for (int i = 0; i < num_pixels; i++) {
        blank_image[i] = color_to_gpu(new color(0, 0, 0));
    }
    hipMemcpy(img_out, blank_image, img_size, hipMemcpyHostToDevice);

    // Assigning all of our variables -- things like camera settings and test triangles
    double fov_scale = 1;
    vector* cam_origin = new vector(0, 0, 0);
    vector* cam_direction = new vector(0, 0, 0);
    camera* main_cam = new camera(cam_origin, cam_direction, fov_scale);

    dimensions* img_dim = new dimensions(width, height);

    int num_tris = 1;
    int size_of_triangles = num_tris * sizeof(triangle*);
    triangle** triangles = new triangle*[num_tris];
    
    material* placeholder_material = new material(new color(255, 255, 255), 1, 0, 0);
    triangle* test_tri_1 = new triangle(placeholder_material, new vector(0, 0, 1), new vector(10, 0, 1), new vector(0, 10, 1));
    triangle* test_gpu = triangle_to_gpu(test_tri_1);
    triangles[0] = test_gpu;

    // Copying all of the variables above to the GPU memory
    camera* gpu_main_cam = camera_to_gpu(main_cam);

    dimensions* gpu_img_dim = dimensions_to_gpu(img_dim);

    triangle** gpu_triangles;
    hipMalloc(&gpu_triangles, size_of_triangles);
    hipMemcpy(gpu_triangles, triangles, size_of_triangles, hipMemcpyHostToDevice);

    // Starting the kernel
    test_kernel<<<
        dim3(1),
        dim3(1),
        0,
        hipStreamDefault
    >>>(img_out, gpu_main_cam, gpu_img_dim, gpu_triangles, num_tris, 0, num_pixels);               // Start index is inclusive, end index is exclusive

    
    // Wait on all active streams on the current device. VERY NECESSARY
    hipDeviceSynchronize();
    
    // Getting our final result from the GPU to the CPU
    color** result = img_to_cpu(img_out, num_pixels);

    return result;
}




JNIEXPORT jdoubleArray JNICALL Java_Main_test(JNIEnv* env, jobject thisObject, jint width, jint height) {
    color** img = run(width, height);

    int num_pixels = width * height;
    int num_colors = num_pixels * 3;
    jdoubleArray img_java = env->NewDoubleArray(num_colors);

    // Creating a usable shallow copy of the img_java array as a jdouble pointer, and filling it with the color values calculated by the GPU to send 
    // back to the Java host program to be displayed
    jboolean* is_copy_ptr = new jboolean[1];
    *is_copy_ptr = JNI_FALSE;
    jdouble* arr_ptrs = env->GetDoubleArrayElements(img_java, is_copy_ptr);              // Getting the img_out jdoubleArray object as a C++ array
    for (int i = 0; i < num_pixels; i++) {
        // Taking the color values from each pixel
        color* curr_color = img[i];
        double r = *curr_color->r;
        double g = *curr_color->g;
        double b = *curr_color->b;

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
