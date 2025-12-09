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
    triangle* tri = new triangle();         // A pointer to the triangle contained within this bounding box -- not sure if this is slow but it's the simplest for now
    
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
        
        x_min = _x_min;
        x_max = _x_max;
        y_min = _y_min;
        y_max = _y_max;
        z_min = _z_min;
        z_max = _z_max;
    }

    ~bounding_box() {
        delete tri;
    }
};


// A single member of a binary tree of bounding boxes used to recursively search through boxes when performing ray-triangle intersection
struct box_node {
    bounding_box box;                   // The box that this node contains
    // Each node has up to two child nodes and a parent node
    box_node* parent = new box_node();
    box_node* child1 = new box_node();
    box_node* child2 = new box_node();
    int num_children;

    // Constructor for a box node with two children
    box_node(box_node _child1, box_node _child2) {
        num_children = 2;
        *child1 = _child1;
        *child2 = _child2;
        
        // Setting this box_node as the parent for its children
        child1->parent = this;
        child2->parent = this;

        box = generate_bounding_box(this);
    }

    // Constructor for a box node with only a single child
    box_node(box_node _child1) {
        num_children = 1;
        *child1 = _child1;

        // Setting this box_node as the parent for its children
        child1->parent = this;

        box = generate_bounding_box(this);
    }

    // Constructor for a box node with no children
    box_node(bounding_box _box) {
        num_children = 0;
        box = _box;
    }

    ~box_node() {
        delete child1;
        delete child2;
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


// Returns a bounding box around the two children of the given box_node
bounding_box generate_bounding_box(box_node node) {
    if (node.num_children == 0) return node.box;
    if (node.num_children == 1) return node.child1->box;

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


// Generates a bounding-box tree using the given triangle array
// Very inefficient until I think up a better solution, but since this is only really going to run once (for now) on a small set of triangles (for 
// now) it's ok -- this is the simplest way I can think of doing this, and I'd like to solve it on my own
box_node generate_bounding_box_tree(triangle* array) {



    return NULL;
}