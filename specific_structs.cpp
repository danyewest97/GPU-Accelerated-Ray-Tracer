// A library file with all of the required more-specific structs and their constructors, plus some methods to use on them -- Mostly structs that are 
// only needed to couple certain specific datatypes together, without any methods to act on them

// A collision class to hold all the necessary information about 
struct collision {
    triangle* collision_triangle;
    color* collision_color;
    vector* collision_point;
    double* collision_distance;
};

// A bounding box struct for creating bounding volume hierarchies (that partition space into volumes to speed up ray-triangle intersection calculation)
// that holds the array index that correspond to the triangle contained within the bounding box
// Basically, kind of like bounding boxes in rasterization: instead of looking at every triangle (or every pixel on screen in the case of 
// rasterization), we split up space into different partitions that each contain multiple triangles and/or multiple other bounding boxes, and then we
// calculate the bounding box-ray intersections (easier and faster than triangles), and if a ray intersects a box, the ray then looks for 
// intersections with only the triangles contained in that box.
// This strategy is MUCH faster because it actually decreases the big-O time complexity of ray-triangle intersection calculation, which is BIG.
// P.S.: Not sure if we should use doubles or ints for the coordinates defining the bounding box, or if it even matters
struct bounding_box {
    int* child_index;                 // The index of the child triangle contained in this bounding box
    
    // The six coordinates defining the bounds of the box (min x, max x, min y, max y, min z, max z), any point between these coordinates is inside 
    // the box and any point not between them is outside
    double* x_min;
    double* x_max;
    double* y_min;
    double* y_max;
    double* z_min;
    double* z_max;

    __device__ bounding_box() {}

    __device__ bounding_box(int* _child_index, double* _x_min, double* _x_max, double* _y_min, double* _y_max, double* _z_min, double* _z_max) {
        child_index = _child_index;
        
        x_min = _x_min;
        x_max = _x_max;
        y_min = _y_min;
        y_max = _y_max;
        z_min = _z_min;
        z_max = _z_max;
    }

    __device__ ~bounding_box() {
        delete child_index, x_min, x_max, y_min, y_max, z_min, z_max;
    }
};
