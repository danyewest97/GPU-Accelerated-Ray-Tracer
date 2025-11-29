// This file has all of the functions for copying different structs over to the GPU in one function call, to simplify initialization of the kernel
// Each function takes in a struct pointer and returns the address of that struct, copied to the GPU with ***all of its members copied as well***

// Each function will follow the same style as this one, but when a struct contains pointers to another struct it will copy that struct to the GPU 
// using its respective copy function -- the vector struct doesn't need to do this though because it only has doubles
__host__ vector* vector_to_gpu(vector* cpu_var) {
    double* gpu_x;
    hipMalloc(&gpu_x, sizeof(double));
    hipMemcpy(gpu_x, cpu_var->x, sizeof(double), hipMemcpyHostToDevice);

    double* gpu_y;
    hipMalloc(&gpu_y, sizeof(double));
    hipMemcpy(gpu_y, cpu_var->y, sizeof(double), hipMemcpyHostToDevice);

    double* gpu_z;
    hipMalloc(&gpu_z, sizeof(double));
    hipMemcpy(gpu_z, cpu_var->z, sizeof(double), hipMemcpyHostToDevice);


    // I just named this gpu_var to make it easier to copy 
    // and paste the function to add all of the different structs.
    // Basically it's a vector on the CPU that has the addresses of its corresponding members on the GPU
    vector* gpu_var = new vector[1];
    gpu_var->x = gpu_x;
    gpu_var->y = gpu_y;
    gpu_var->z = gpu_z;

    vector* result;
    hipMalloc(&result, sizeof(vector));
    hipMemcpy(result, gpu_var, sizeof(vector), hipMemcpyHostToDevice);

    return result;
}


__host__ color* color_to_gpu(color* cpu_var) {
    double* gpu_r;
    hipMalloc(&gpu_r, sizeof(double));
    hipMemcpy(gpu_r, cpu_var->r, sizeof(double), hipMemcpyHostToDevice);

    double* gpu_g;
    hipMalloc(&gpu_g, sizeof(double));
    hipMemcpy(gpu_g, cpu_var->g, sizeof(double), hipMemcpyHostToDevice);

    double* gpu_b;
    hipMalloc(&gpu_b, sizeof(double));
    hipMemcpy(gpu_b, cpu_var->b, sizeof(double), hipMemcpyHostToDevice);


    color* gpu_var = new color[1];
    gpu_var->r = gpu_r;
    gpu_var->g = gpu_g;
    gpu_var->b = gpu_b;

    color* result;
    hipMalloc(&result, sizeof(color));
    hipMemcpy(result, gpu_var, sizeof(color), hipMemcpyHostToDevice);

    return result;
}


__host__ material* material_to_gpu(material* cpu_var) {
    color* gpu_material_color = color_to_gpu(cpu_var->material_color);

    double* gpu_diffusion;
    hipMalloc(&gpu_diffusion, sizeof(double));
    hipMemcpy(gpu_diffusion, cpu_var->diffusion, sizeof(double), hipMemcpyHostToDevice);

    double* gpu_reflection;
    hipMalloc(&gpu_reflection, sizeof(double));
    hipMemcpy(gpu_reflection, cpu_var->reflection, sizeof(double), hipMemcpyHostToDevice);

    double* gpu_refraction;
    hipMalloc(&gpu_refraction, sizeof(double));
    hipMemcpy(gpu_refraction, cpu_var->refraction, sizeof(double), hipMemcpyHostToDevice);


    material* gpu_var = new material[1];
    gpu_var->material_color = gpu_material_color;
    gpu_var->diffusion = gpu_diffusion;
    gpu_var->reflection = gpu_reflection;
    gpu_var->refraction = gpu_refraction;

    material* result;
    hipMalloc(&result, sizeof(material));
    hipMemcpy(result, gpu_var, sizeof(material), hipMemcpyHostToDevice);

    return result;
}


__host__ plane* plane_to_gpu(plane* cpu_var) {
    vector* gpu_normal = vector_to_gpu(cpu_var->normal);

    double* gpu_d;
    hipMalloc(&gpu_d, sizeof(double));
    hipMemcpy(gpu_d, cpu_var->d, sizeof(double), hipMemcpyHostToDevice);


    plane* gpu_var = new plane[1];
    gpu_var->normal = gpu_normal;
    gpu_var->d = gpu_d;

    plane* result;
    hipMalloc(&result, sizeof(plane));
    hipMemcpy(result, gpu_var, sizeof(plane), hipMemcpyHostToDevice);

    return result;
}


__host__ ray* ray_to_gpu(ray* cpu_var) {
    vector* gpu_origin = vector_to_gpu(cpu_var->origin);

    vector* gpu_direction = vector_to_gpu(cpu_var->direction);


    ray* gpu_var = new ray[1];
    gpu_var->origin = gpu_origin;
    gpu_var->direction = gpu_direction;

    ray* result;
    hipMalloc(&result, sizeof(ray));
    hipMemcpy(result, gpu_var, sizeof(ray), hipMemcpyHostToDevice);

    return result;
}


__host__ triangle* triangle_to_gpu(triangle* cpu_var) {
    plane* gpu_surface_plane = plane_to_gpu(cpu_var->surface_plane);

    material* gpu_surface_material = material_to_gpu(cpu_var->surface_material);

    vector* gpu_a = vector_to_gpu(cpu_var->a);

    vector* gpu_b = vector_to_gpu(cpu_var->b);

    vector* gpu_c = vector_to_gpu(cpu_var->c);


    triangle* gpu_var = new triangle[1];
    gpu_var->surface_plane = gpu_surface_plane;
    gpu_var->surface_material = gpu_surface_material;
    gpu_var->a = gpu_a;
    gpu_var->b = gpu_b;
    gpu_var->c = gpu_c;

    triangle* result;
    hipMalloc(&result, sizeof(triangle));
    hipMemcpy(result, gpu_var, sizeof(triangle), hipMemcpyHostToDevice);

    return result;
}


__host__ dimensions* dimensions_to_gpu(dimensions* cpu_var) {;
    int* gpu_width;
    hipMalloc(&gpu_width, sizeof(int));
    hipMemcpy(gpu_width, cpu_var->width, sizeof(int), hipMemcpyHostToDevice);

    int* gpu_height;
    hipMalloc(&gpu_height, sizeof(int));
    hipMemcpy(gpu_height, cpu_var->height, sizeof(int), hipMemcpyHostToDevice);


    dimensions* gpu_var = new dimensions[1];
    gpu_var->width = gpu_width;
    gpu_var->height = gpu_height;

    dimensions* result;
    hipMalloc(&result, sizeof(dimensions));
    hipMemcpy(result, gpu_var, sizeof(dimensions), hipMemcpyHostToDevice);

    return result;
}


__host__ camera* camera_to_gpu(camera* cpu_var) { 
    vector* gpu_origin = vector_to_gpu(cpu_var->origin);
    
    vector* gpu_rotation = vector_to_gpu(cpu_var->rotation);
    
    double* gpu_fov_scale;
    hipMalloc(&gpu_fov_scale, sizeof(double));
    hipMemcpy(gpu_fov_scale, cpu_var->fov_scale, sizeof(double), hipMemcpyHostToDevice);

    camera* gpu_var = new camera[1];
    gpu_var->origin = gpu_origin;
    gpu_var->rotation = gpu_rotation;
    gpu_var->fov_scale = gpu_fov_scale;

    camera* result;
    hipMalloc(&result, sizeof(camera));
    hipMemcpy(result, gpu_var, sizeof(camera), hipMemcpyHostToDevice);

    return result;
}


__host__ light* light_to_gpu(light* cpu_var) {
    vector* gpu_position = vector_to_gpu(cpu_var->position);

    color* gpu_rgb = color_to_gpu(cpu_var->rgb);

    double* gpu_intensity;
    hipMalloc(&gpu_intensity, sizeof(double));
    hipMemcpy(gpu_intensity, cpu_var->intensity, sizeof(double), hipMemcpyHostToDevice);


    light* gpu_var = new light[1];
    gpu_var->position = gpu_position;
    gpu_var->rgb = gpu_rgb;
    gpu_var->intensity = gpu_intensity;

    light* result;
    hipMalloc(&result, sizeof(light));
    hipMemcpy(result, gpu_var, sizeof(light), hipMemcpyHostToDevice);

    return result;
}



// These are the functions that do the opposite of the above functions -- they take a pointer to a variable stored on the GPU and return a pointer to 
// the same variable copied to the CPU with all of its members. This will need to be implemented for every struct as well eventually, but for now I am 
// only doing the color struct in order to get an output image
__host__ color* color_to_cpu(color* gpu_var) {
    color* cpu_var = new color[1];
    hipMemcpy(cpu_var, gpu_var, sizeof(color), hipMemcpyDeviceToHost);                      // Copying the GPU addresses over so that we can access 
    // them in the following lines, on the CPU
    
    double* cpu_r = new double[1];
    hipMemcpy(cpu_r, cpu_var->r, sizeof(double), hipMemcpyDeviceToHost);

    double* cpu_g = new double[1];
    hipMemcpy(cpu_g, cpu_var->g, sizeof(double), hipMemcpyDeviceToHost);

    double* cpu_b = new double[1];
    hipMemcpy(cpu_b, cpu_var->b, sizeof(double), hipMemcpyDeviceToHost);

    cpu_var->r = cpu_r;
    cpu_var->g = cpu_g;
    cpu_var->b = cpu_b;

    return cpu_var;
}


// This function is specially used to take the output image and transfer it to the CPU, so it needs to take in a size argument as well that 
// corresponds to the number of pixels
__host__ color** img_to_cpu(color** gpu_var, int size) {
    color** cpu_var = new color*[size];
    hipMemcpy(cpu_var, gpu_var, sizeof(color*) * size, hipMemcpyDeviceToHost);              // Copying the GPU addresses over so that we can access 
                                                                                            // them in the following lines, on the CPU
    
    
    for (int i = 0; i < size; i++) {
        color* cpu_color = color_to_cpu(cpu_var[i]);
        cpu_var[i] = cpu_color;
    }
    
    return cpu_var;
}
