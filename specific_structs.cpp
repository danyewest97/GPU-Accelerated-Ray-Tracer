// A library file with all of the required more-specific structs and their constructors, plus some methods to use on them -- Mostly structs that are 
// only needed to couple certain specific datatypes together, without any methods to act on them

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
    triangle tri;                          // A pointer to the triangle contained within this bounding box -- not sure if this is slow but it's the 
                                            // simplest for now
                                            // This argument is OPTIONAL because a bounding box could represent the bounds of two or more other 
                                            // bounding boxes/triangles, instead of just a single triangle, which is used in the box_node struct
    
    // The six coordinates defining the bounds of the box (min x, max x, min y, max y, min z, max z), any point between these coordinates is inside 
    // the box and any point not between them is outside
    double min_x;
    double max_x;
    double min_y;
    double max_y;
    double min_z;
    double max_z;

    bounding_box() {}

    bounding_box(triangle _tri, double _min_x, double _max_x, double _min_y, double _max_y, double _min_z, double _max_z) {
        tri = _tri;
        represents_single_triangle = true;

        min_x = _min_x;
        max_x = _max_x;
        min_y = _min_y;
        max_y = _max_y;
        min_z = _min_z;
        max_z = _max_z;
    }

    // Constructor without a tri argument, used when this box represents the bounds of multiple other boxes or triangles
    __host__ __device__ bounding_box(double _min_x, double _max_x, double _min_y, double _max_y, double _min_z, double _max_z) {
        min_x = _min_x;
        max_x = _max_x;
        min_y = _min_y;
        max_y = _max_y;
        min_z = _min_z;
        max_z = _max_z;
    }
};




// Returns a bounding box around the given triangle
bounding_box generate_bounding_box(triangle tri) {
    std::vector<double> x_vals = {tri.a.x, tri.b.x, tri.c.x};
    std::vector<double> y_vals = {tri.a.y, tri.b.y, tri.c.y};
    std::vector<double> z_vals = {tri.a.z, tri.b.z, tri.c.z};
    
    std::sort(x_vals.begin(), x_vals.end());
    std::sort(y_vals.begin(), y_vals.end());
    std::sort(z_vals.begin(), z_vals.end());
    
    double min_x = x_vals.front();
    double min_y = y_vals.front();
    double min_z = z_vals.front();

    double max_x = x_vals.back();
    double max_y = y_vals.back();
    double max_z = z_vals.back();

    bounding_box result(tri, min_x, max_x, min_y, max_y, min_z, max_z);
    return result;
}


// Declaring the box_node type and generate_bounding_box() method so that we can use generate_bounding_box(), which takes a box_node argument, in 
// the box_node constructor.
// Kind of a hack imo but it's the easiest way to get around to doing this, without having to put the method at the top of the box_node struct
// definition, before its constructors.

// A single member of an octree of bounding boxes used to recursively search through boxes when performing ray-triangle intersection
struct box_node {
    bounding_box box;                   // The box that this node contains
    // Each node has up to eight child nodes and a parent node
    box_node* parent;
    int num_children = 8;
    box_node** children = new box_node*[num_children];
    bool is_empty = false;

    box_node() {}

    box_node(int _num_children) {
        num_children = _num_children;
        // children = new box_node*[num_children];
    }

    // Constructor for a base box node node with no children -- a "leaf" in the tree of nodes
    box_node(bounding_box _box) {
        num_children = 0;
        box = _box;
    }
};


struct centroid {
    vector pos;
    bounding_box box;

    centroid() {}

    centroid(vector _pos, bounding_box _box) {
        pos = _pos;
        box = _box;
    }
};


// Returns a bounding box around the children of the given box_node
bounding_box generate_bounding_box(box_node node) {
    int num_children = node.num_children;
    if (num_children == 0) return node.box;

    std::vector<double> x_vals;
    std::vector<double> y_vals;
    std::vector<double> z_vals;
    
    for (int i = 0; i < num_children; i++) {
        bounding_box curr_box = node.children[i]->box;
        
        x_vals.push_back(curr_box.min_x);
        x_vals.push_back(curr_box.max_x);

        y_vals.push_back(curr_box.min_y);
        y_vals.push_back(curr_box.max_y);

        z_vals.push_back(curr_box.min_z);
        z_vals.push_back(curr_box.max_z);
    }
    
    
    std::sort(x_vals.begin(), x_vals.end());
    std::sort(y_vals.begin(), y_vals.end());
    std::sort(z_vals.begin(), z_vals.end());

    double min_x = x_vals.front();
    double min_y = y_vals.front();
    double min_z = z_vals.front();

    double max_x = x_vals.back();
    double max_y = y_vals.back();
    double max_z = z_vals.back();
    
    bounding_box result(min_x, max_x, min_y, max_y, min_z, max_z);
    return result;
}


// Returns a bounding box around the given boxes
bounding_box generate_bounding_box(centroid* centers, int num_centers) {
    int num_values_per_center = 2;
    int num_values = num_centers * num_values_per_center;
    
    std::vector<double> x_vals;
    std::vector<double> y_vals;
    std::vector<double> z_vals;
    
    for (int i = 0; i < num_centers; i++) {
        bounding_box curr_box = centers[i].box;
        
        x_vals.push_back(curr_box.min_x);
        x_vals.push_back(curr_box.max_x);

        y_vals.push_back(curr_box.min_y);
        y_vals.push_back(curr_box.max_y);

        z_vals.push_back(curr_box.min_z);
        z_vals.push_back(curr_box.max_z);
    }
    
    std::sort(x_vals.begin(), x_vals.end());
    std::sort(y_vals.begin(), y_vals.end());
    std::sort(z_vals.begin(), z_vals.end());
    
    double min_x = x_vals.front();
    double min_y = y_vals.front();
    double min_z = z_vals.front();

    double max_x = x_vals.back();
    double max_y = y_vals.back();
    double max_z = z_vals.back();

    bounding_box result(min_x, max_x, min_y, max_y, min_z, max_z);
    return result;
}




centroid* generate_centroids(triangle* tris, int num_tris) {
    centroid* centers = new centroid[num_tris];
    
    for (int i = 0; i < num_tris; i++) {
        vector center(0, 0, 0);
        triangle tri = tris[i];
        center.add(tri.a);
        center.add(tri.b);
        center.add(tri.c);
        center.mult(1.0 / 3.0);
        bounding_box box = generate_bounding_box(tri);
        centers[i] = centroid(center, box);
    }
    
    return centers;
}


vector get_midpoint(centroid* centers, int num_tris) {
    vector midpoint(0, 0, 0);
    
    for (int i = 0; i < num_tris; i++) {
        midpoint.add(centers[i].pos);
    }
    midpoint.mult(1.0 / num_tris);
    
    return midpoint;
}

centroid* tris_in_quadrant(int quadrant, vector midpoint, centroid* centers, int num_tris_in, int* num_tris_out) {
    int num_contained = 0;
    int* contained_center_indices = new int[num_tris_in];

    std::bitset<3> direction_coefficients(quadrant);                // Turns our search quadrant into binary representation which tells us where we 
                                                                    // need to look -- 0 == look in the negative direction, 1 == look in the 
                                                                    // positive direction (i.e. if quadrant == 1, coefficients == 001, meaning we
                                                                    // look in the negative y- and z-directions and the positive x-direction)
                                                                    // NOTE: bits read right-to-left of normal representation, so if coefficients == 
                                                                    // 001 like above, coefficients[0] == 1, coefficients[1] == 0, and
                                                                    // coefficients[2] == 0
    
    bool x_search = direction_coefficients[0];
    bool y_search = direction_coefficients[1];
    bool z_search = direction_coefficients[2];

    for (int i = 0; i < num_tris_in; i++) {
        centroid center = centers[i];
        vector center_pos = center.pos;
        
        bool contained_x = center_pos.x >= midpoint.x;
        bool contained_y = center_pos.y >= midpoint.y;
        bool contained_z = center_pos.z >= midpoint.z;

        if (x_search) contained_x = !contained_x;
        if (y_search) contained_y = !contained_y;
        if (z_search) contained_z = !contained_z;

        if (contained_x && contained_y && contained_z) {
            contained_center_indices[num_contained] = i;
            num_contained++;
        }
    }

    centroid* result = new centroid[num_contained];
    for (int i = 0; i < num_contained; i++) {
        int contained_idx = contained_center_indices[i];
        result[i] = centroid(centers[contained_idx].pos, centers[contained_idx].box);
    }

    *num_tris_out = num_contained;
    
    return result;
}


box_node* get_child_node(box_node* parent, centroid* centers, vector midpoint, int num_tris, int quadrant) {
    int num_contained = 0;
    centroid* new_centers = tris_in_quadrant(quadrant, midpoint, centers, num_tris, &num_contained);
    
    if (num_contained == 0) {
        box_node* result = new box_node(0);
        result->parent = parent;
        result->is_empty = true;
        return result;
    }

    if (num_contained <= 8 || num_contained == num_tris) {
        box_node* result = new box_node(num_contained);
        result->parent = parent;
        bounding_box box = generate_bounding_box(new_centers, num_contained);
        result->box = box;
        for (int i = 0; i < num_contained; i++) {
            box_node* child = new box_node(new_centers[i].box);
            child->parent = result;
            result->children[i] = child;
        }
        return result;
    }

    vector new_midpoint = get_midpoint(new_centers, num_contained);

    int num_quadrants = 8;
    box_node* result = new box_node(num_quadrants);
    for (int i = 0; i < num_quadrants; i++) {
        result->children[i] = get_child_node(result, new_centers, new_midpoint, num_contained, i);
    }
    return result;
}

box_node* generate_box_tree(triangle* tris, int num_tris) {
    box_node* root = new box_node();
    centroid* centers = generate_centroids(tris, num_tris);
    
    vector midpoint = get_midpoint(centers, num_tris);

    int num_quadrants = 8;
    for (int i = 0; i < num_quadrants; i++) {
        root->children[i] = get_child_node(root, centers, midpoint, num_tris, i);
    }

    return root;
}






__device__ bool ray_box_intersection(ray& r, bounding_box& box) {
    vector& direction = r.direction;
    vector& origin = r.origin;

    if (r.direction.x != 0) {
        double left = direction.x;
        double right = origin.x + box.min_x;
        double t = right / left;
        
        double y = direction.y * t + origin.y;
        double z = direction.z * t + origin.z;
        
        if (t >= 0) {
            if (y >= box.min_y && y <= box.max_y) {
                if (z >= box.min_z && z <= box.max_z) {
                    return true;
                }
            }
        }
    }


    if (r.direction.y != 0) {
        double left = direction.y;
        double right = origin.y + box.min_y;
        double t = right / left;
        
        double z = direction.z * t + origin.z;
        double x = direction.x * t + origin.x;

        if (t >= 0) {
            if (z >= box.min_z && z <= box.max_z) {
                if (x >= box.min_x && x <= box.max_x) {
                    return true;
                }
            }
        }
    }


    if (r.direction.z != 0) {
        double left = direction.z;
        double right = origin.z + box.min_z;
        double t = right / left;
        
        double x = direction.x * t + origin.x;
        double y = direction.y * t + origin.y;

        if (t >= 0) {
            if (x >= box.min_x && x <= box.max_x) {
                if (y >= box.min_y && y <= box.max_y) {
                    return true;
                }
            }
        }
    }

    return false;
}


// Checks for a ray-triangle intersection given the root node of a tree of bounding boxes
__device__ hit ray_triangle_intersection(ray& r, box_node* root_node) {
    if (root_node->is_empty) {
        return hit();
    }

    if (root_node->num_children == 0) {
        if (ray_box_intersection(r, root_node->box)) {
            hit tri_hit = ray_triangle_intersection(r, root_node->box.tri);
            return tri_hit;
        }
        return hit();
    }

    hit closest_hit;
    bool has_any_intersection = false;

    for (int i = 0; i < root_node->num_children; i++) {
        box_node* curr_child = root_node->children[i];
        if (curr_child->is_empty || !ray_box_intersection(r, curr_child->box)) continue;

        hit curr_hit = ray_triangle_intersection(r, curr_child);
        
        if (curr_hit.has_intersection && !has_any_intersection) {
            closest_hit = curr_hit;
        } else if (curr_hit.has_intersection) {
            if (curr_hit.hit_distance < closest_hit.hit_distance) {
                closest_hit = curr_hit;
            }
        }

        has_any_intersection |= curr_hit.has_intersection;
    }

    if (has_any_intersection) {
        return closest_hit;
    }
    return hit();
}


// Checks for a ray-triangle intersection given an array of triangles and a given triangle to exclude from the search (i.e. if the ray bounced off a 
// triangle, we don't want to count that triangle again or the ray could intersect with it twice in a row due to precision errors, causing weird 
// artifacts)
// __device__ vector ray_triangle_intersection(ray r, triangle* tris, bool* has_intersection, double* t_out) {

// }







box_node* prepare_box_node_for_gpu(box_node* cpu_node) {
    if (cpu_node->num_children == 0) {
        box_node* gpu_node;
        hipMalloc(&gpu_node, sizeof(box_node));
        hipMemcpy(gpu_node, cpu_node, sizeof(box_node), hipMemcpyHostToDevice);
        return gpu_node;
    }

    for (int i = 0; i < cpu_node->num_children; i++) {
        cpu_node->children[i] = prepare_box_node_for_gpu(cpu_node->children[i]);
    }


    int size_of_children = sizeof(box_node) * cpu_node->num_children;
    box_node** gpu_children;
    hipMalloc(&gpu_children, size_of_children);
    hipMemcpy(gpu_children, cpu_node->children, size_of_children, hipMemcpyHostToDevice);
    cpu_node->children = gpu_children;


    int size_of_node = size_of_children + sizeof(box_node);
    box_node* gpu_node;
    hipMalloc(&gpu_node, size_of_node);
    hipMemcpy(gpu_node, cpu_node, size_of_node, hipMemcpyHostToDevice);
    return gpu_node;
}







