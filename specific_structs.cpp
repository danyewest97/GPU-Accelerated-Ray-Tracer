// A library file with all of the required more-specific structs and their constructors, plus some methods to use on them -- Mostly structs that are 
// only needed to couple certain specific datatypes together, without any methods to act on them

// A hit class to hold all the necessary information about 
struct hit {
    triangle hit_triangle;
    color hit_color;
    vector hit_point;
    double hit_distance;

    __device__ hit() {}

    __device__ hit(triangle _hit_triangle, color _hit_color, vector _hit_point, double _hit_distance) {
        hit_triangle = _hit_triangle;
        hit_color = _hit_color;
        hit_point = _hit_point;
        hit_distance = _hit_distance;
    }
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
    bool represents_single_triangle = false;// Whether or not this bounding box represents a single triangle
    triangle* tri = new triangle();         // A pointer to the triangle contained within this bounding box -- not sure if this is slow but it's the 
                                            // simplest for now
                                            // This argument is OPTIONAL because a bounding box could represent the bounds of two or more other 
                                            // bounding boxes/triangles, instead of just a single triangle, which is used in the box_node struct
    
    // The six coordinates defining the bounds of the box (min x, max x, min y, max y, min z, max z), any point between these coordinates is inside 
    // the box and any point not between them is outside
    double x_min;
    double x_max;
    double y_min;
    double y_max;
    double z_min;
    double z_max;

    bounding_box() {}

    bounding_box(triangle _tri, double _x_min, double _x_max, double _y_min, double _y_max, double _z_min, double _z_max) {
        *tri = _tri;
        represents_single_triangle = true;

        x_min = _x_min;
        x_max = _x_max;
        y_min = _y_min;
        y_max = _y_max;
        z_min = _z_min;
        z_max = _z_max;
    }

    // Constructor without a tri argument, used when this box represents the bounds of multiple other boxes or triangles
    bounding_box(double _x_min, double _x_max, double _y_min, double _y_max, double _z_min, double _z_max) {
        x_min = _x_min;
        x_max = _x_max;
        y_min = _y_min;
        y_max = _y_max;
        z_min = _z_min;
        z_max = _z_max;
    }
};




// Returns a bounding box around the given triangle
bounding_box generate_bounding_box(triangle tri) {
    // Manually finding the minimum and maximum x, y, and z values to create a bounding box around the given triangle
    double min_x = tri.a.x;
    double min_y = tri.a.y;
    double min_z = tri.a.z;


    if (tri.b.x < min_x) min_x = tri.b.x;
    if (tri.c.x < min_x) min_x = tri.c.x;

    if (tri.b.y < min_y) min_y = tri.b.y;
    if (tri.c.y < min_y) min_y = tri.c.y;

    if (tri.b.z < min_z) min_z = tri.b.z;
    if (tri.c.z < min_z) min_z = tri.c.z;



    double max_x = tri.a.x;
    double max_y = tri.a.y;
    double max_z = tri.a.z;


    if (tri.b.x > max_x) max_x = tri.b.x;
    if (tri.c.x > max_x) max_x = tri.c.x;

    if (tri.b.y > max_y) max_y = tri.b.y;
    if (tri.c.y > max_y) max_y = tri.c.y;

    if (tri.b.z > max_z) max_z = tri.b.z;
    if (tri.c.z > max_z) max_z = tri.c.z;


    bounding_box result(tri, min_x, max_x, min_y, max_y, min_z, max_z);
    return result;
}


// Declaring the box_node type and generate_bounding_box() method so that we can use generate_bounding_box(), which takes a box_node argument, in 
// the box_node constructor.
// Kind of a hack imo but it's the easiest way to get around to doing this, without having to put the method at the top of the box_node struct d
// efinition, before its constructors.
struct box_node;
bounding_box generate_bounding_box(box_node node);

// A single member of a binary tree of bounding boxes used to recursively search through boxes when performing ray-triangle intersection
struct box_node {
    bounding_box box;                   // The box that this node contains
    // Each node has up to two child nodes and a parent node
    box_node* parent;
    box_node* child1;
    box_node* child2;
    int num_children;
    

    box_node() {}

    // Constructor for a box node with two children
    box_node(box_node* _child1, box_node* _child2) {
        num_children = 2;
        child1 = _child1;
        child2 = _child2;
        
        // Setting this box_node as the parent for its children
        child1->parent = this;
        child2->parent = this;

        // box = generate_bounding_box(this);
    }

    // Constructor for a box node with only a single child
    box_node(box_node* _child1) {
        num_children = 1;
        child1 = _child1;

        // Setting this box_node as the parent for its children
        child1->parent = this;

        box = generate_bounding_box(this);
    }

    // Constructor for a box node with no children
    box_node(bounding_box _box) {
        num_children = 0;
        box = _box;
    }
};


// Returns a bounding box around the two children of the given box_node
bounding_box generate_bounding_box(box_node node) {
    if (node.num_children == 0) return node.box;
    if (node.num_children == 1) return node.child1->box;

    // Note: May need to simplify what is a pointer and what is not, because this... looks horrendous with the arrows and dots interchanging
    triangle tri1 = *(node.child1->box.tri);
    triangle tri2 = *(node.child2->box.tri);

    double x_values[6] = {tri1.a.x, tri1.b.x, tri1.c.x,
                        tri2.a.x, tri2.b.x, tri2.c.x};
    double y_values[6] = {tri1.a.y, tri1.b.y, tri1.c.y,
                        tri2.a.y, tri2.b.y, tri2.c.y};
    double z_values[6] = {tri1.a.z, tri1.b.z, tri1.c.z,
                        tri2.a.z, tri2.b.z, tri2.c.z};

    // Manually finding the minimum and maximum x, y, and z values to create a bounding box around the given triangle
    double min_x = x_values[0];
    double min_y = y_values[0];
    double min_z = z_values[0];

    for (int i = 1; i < 6; i++) {
        if (x_values[i] < min_x) min_x = x_values[i];
        if (y_values[i] < min_y) min_y = y_values[i];
        if (z_values[i] < min_z) min_z = z_values[i];
    }


    double max_x = x_values[0];
    double max_y = y_values[0];
    double max_z = z_values[0];


    for (int i = 1; i < 6; i++) {
        if (x_values[i] > max_x) max_x = x_values[i];
        if (y_values[i] > max_y) max_y = y_values[i];
        if (z_values[i] > max_z) max_z = z_values[i];
    }


    bounding_box result(min_x, max_x, min_y, max_y, min_z, max_z);
    return result;
}



// Generates a bounding-box tree using the given triangle array
// Very inefficient until I think up a better solution, but since this is only really going to run once (for now) on a small set of triangles (for 
// now) it's ok -- this is the simplest way I can think of doing this, and I'd like to solve it on my own
box_node generate_bounding_box_tree(triangle* array, int num_tris) {
    


    return NULL;
}