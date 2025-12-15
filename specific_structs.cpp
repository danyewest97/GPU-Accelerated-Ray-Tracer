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
    bounding_box(double _min_x, double _max_x, double _min_y, double _max_y, double _min_z, double _max_z) {
        min_x = _min_x;
        max_x = _max_x;
        min_y = _min_y;
        max_y = _max_y;
        min_z = _min_z;
        max_z = _max_z;
    }
};




// Returns a bounding box around the given triangle
bounding_box generate_bounding_box(triangle& tri) {
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
// Kind of a hack imo but it's the easiest way to get around to doing this, without having to put the method at the top of the box_node struct
// definition, before its constructors.

// A single member of an octree of bounding boxes used to recursively search through boxes when performing ray-triangle intersection
struct box_node {
    bounding_box box;                   // The box that this node contains
    // Each node has up to eight child nodes and a parent node
    int num_children = 8;
    box_node* children;
    

    box_node() {}

    // Copy constructor
    // box_node(box_node& old_node) {
    //     box = old_node.box;
    //     num_children = old_node.num_children;
    //     children = new box_node[num_children];
    //     for (int i = 0; i < num_children; i++) {
    //         children[i] = old_node.children[i];
    //     }
    // }

    box_node(int _num_children) {
        num_children = _num_children;
        children = new box_node[num_children];
    }

    // Constructor for a base box node node with no children -- a "leaf" in the tree of nodes
    box_node(bounding_box _box) {
        num_children = 0;
        box = _box;
    }
};



// Returns a bounding box around the children of the given box_node
bounding_box generate_bounding_box(box_node& node) {
    int num_children = node.num_children;
    if (num_children == 0) return node.box;

    int num_values_per_child = 2;
    int num_raw_values = num_children * num_values_per_child;
    
    double* x_values = new double[num_raw_values];
    double* y_values = new double[num_raw_values];
    double* z_values = new double[num_raw_values];
    
    for (int i = 0; i < num_children; i++) {
        bounding_box curr_box = node.children[i].box;
        
        int num_values_per_box = 2;
        int value_idx = i * num_values_per_box;
        
        x_values[value_idx] = curr_box.min_x;
        x_values[value_idx + 1] = curr_box.max_x;
        
        y_values[value_idx] = curr_box.min_z;
        y_values[value_idx + 1] = curr_box.max_y;
        
        z_values[value_idx] = curr_box.min_z;
        z_values[value_idx + 1] = curr_box.max_z;
    }
    
    
    // Manually finding the minimum and maximum x, y, and z values to create a bounding box around the given triangle
    double min_x = x_values[0];
    double min_y = y_values[0];
    double min_z = z_values[0];
    
    for (int i = 1; i < num_raw_values; i++) {
        if (x_values[i] < min_x) min_x = x_values[i];
        if (y_values[i] < min_y) min_y = y_values[i];
        if (z_values[i] < min_z) min_z = z_values[i];
    }
    
    
    double max_x = x_values[0];
    double max_y = y_values[0];
    double max_z = z_values[0];

    
    for (int i = 1; i < num_raw_values; i++) {
        if (x_values[i] > max_x) max_x = x_values[i];
        if (y_values[i] > max_y) max_y = y_values[i];
        if (z_values[i] > max_z) max_z = z_values[i];
    }
    
    
    bounding_box result(min_x, max_x, min_y, max_y, min_z, max_z);
    return result;
}



struct centroid {
    vector center;
    triangle parent;

    centroid() {}

    centroid(vector& _center, triangle& _parent) {
        center = _center;
        parent = _parent;
    }
};

// Recursively creates a child bounding_box node taking the given parent node
// The search quadrant defines which direction of the center point we are looking for triangles (like the quadrants of a 2D graph but in 3D)
// Basically we look at all of the centroids in the given direction of the centerpoint and generate a bounding-box tree for them recursively.
// Search quadrant can be 0-7, which is which doesn't really matter because all will be called consecutively any time this function is in use
// (Octant == quadrant but with 8 instead of 4, i.e. in 2D vs. 3D, where an intersection of 3 planes makes 8 regions as opposed to 4 with 2 lines)
box_node generate_child_node(vector centerpoint, int search_octant, centroid* centroids, int num_search_tris) {
    // Breaking out of our recursion, aka the base case where we have 8 or less centroids/triangles left in the octant and we return a node with the 
    // bounding boxes around each triangle as its children
    if (num_search_tris <= 8) {
        box_node result(num_search_tris);
        for (int i = 0; i < num_search_tris; i++) {
            triangle curr_tri = centroids[i].parent;
            bounding_box tri_box = generate_bounding_box(curr_tri);
            box_node single_tri_node(tri_box);
            result.children[i] = single_tri_node;
        }
        result.box = generate_bounding_box(result);
        return result;
    }
    
    
    vector search_direction;

    if (search_octant == 0) search_direction = vector(1, 1, 1);
    if (search_octant == 1) search_direction = vector(1, 1, -1);
    if (search_octant == 2) search_direction = vector(1, -1, 1);
    if (search_octant == 3) search_direction = vector(1, -1, -1);
    if (search_octant == 4) search_direction = vector(-1, 1, 1);
    if (search_octant == 5) search_direction = vector(-1, 1, -1);
    if (search_octant == 6) search_direction = vector(-1, -1, 1);
    if (search_octant == 7) search_direction = vector(-1, -1, -1);

    vector curr_octant_centerpoint(0, 0, 0);
    double* contained_centroid_indices = new double[num_search_tris];
    int num_centroids_contained = 0;
    for (int i = 0; i < num_search_tris; i++) {
        centroid curr_centroid = centroids[i];
        vector curr_centroid_position = curr_centroid.center;
        double x = curr_centroid_position.x * search_direction.x;
        double y = curr_centroid_position.y * search_direction.y;
        double z = curr_centroid_position.z * search_direction.z;
        
        if (x <= centerpoint.x && y <= centerpoint.y && z <= centerpoint.z) {
            contained_centroid_indices[num_centroids_contained] = i;
            num_centroids_contained++;
            
            curr_octant_centerpoint.add(curr_centroid_position);
        }
    }
    
    curr_octant_centerpoint.mult(1.0 / num_centroids_contained);
    
    centroid* new_centroids = new centroid[num_centroids_contained];
    for (int i = 0; i < num_centroids_contained; i++) {
        int idx = contained_centroid_indices[i];
        new_centroids[i] = centroids[idx];
    }
    

    // Edge case for if we didn't eliminate any triangles when searching (i.e. if there are 9 or more identical triangles)
    if (num_centroids_contained == num_search_tris) {
        box_node result(num_search_tris);
        for (int i = 0; i < num_search_tris; i++) {
            triangle curr_tri = centroids[i].parent;
            bounding_box tri_box = generate_bounding_box(curr_tri);
            box_node single_tri_node(tri_box);
            result.children[i] = single_tri_node;
        }
        result.box = generate_bounding_box(result);
        return result;
    }


    box_node result(8);


    // Creating a recursive sequence that goes through and further adds to the final tree
    for (int i = 0; i < 8; i++) {
        result.children[i] = generate_child_node(curr_octant_centerpoint, i, new_centroids, num_centroids_contained);
    }


    // Freeing memory so we don't get a memory leak
    // delete[] contained_centroid_indices;
    // delete[] new_centroids;

    return result;
}


// Generates a bounding-box tree using the given triangle array
// Probably inefficient, I can clean up later but I just need it to work for now, especially because this likely won't be a bottleneck anytime soon
box_node* generate_bounding_box_tree(triangle* tris, int num_tris) {
    centroid* centroids = new centroid[num_tris];
    for (int i = 0; i < num_tris; i++) {
        triangle curr_tri = tris[i];

        vector sum_of_vertices(curr_tri.a.x + curr_tri.b.x + curr_tri.c.x,
                               curr_tri.a.y + curr_tri.b.y + curr_tri.c.y,
                               curr_tri.a.z + curr_tri.b.z + curr_tri.c.z);
        sum_of_vertices.mult(1.0 / 3);
        centroids[i] = centroid(sum_of_vertices, curr_tri);
    }

    vector midpoint_of_all_vertices(0, 0, 0);
    for (int i = 0; i < num_tris; i++) {
        centroid curr_centroid = centroids[i];
        vector curr_centroid_position = curr_centroid.center;
        midpoint_of_all_vertices.add(curr_centroid_position);
    }
    midpoint_of_all_vertices.mult(1.0 / num_tris);

    box_node* root_node = new box_node(8);
    for (int i = 0; i < 8; i++) {
        box_node child = generate_child_node(midpoint_of_all_vertices, i, centroids, num_tris);
        root_node->children[i] = child;
    }

    return root_node;
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
    vector& direction = r.direction;
    vector& origin = r.origin;

    if (root_node->num_children == 0) {
        hit tri_hit = ray_triangle_intersection(r, root_node->box.tri);
        return tri_hit;
    }

    hit closest_hit;
    bool has_any_intersection = false;

    for (int i = 0; i < root_node->num_children; i++) {
        box_node* curr_child = &(root_node->children[i]);
        if (!ray_box_intersection(r, curr_child->box)) continue;

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
    bool is_second_to_last_node = true;
    for (int i = 0; i < cpu_node->num_children; i++) {
        box_node* curr_child = &(cpu_node->children[i]);
        if (!curr_child->num_children == 0) {
            is_second_to_last_node = false;
            curr_child = prepare_box_node_for_gpu(curr_child);
        }
    }

    if (is_second_to_last_node) {
        int size_of_children = sizeof(box_node) * cpu_node->num_children;
        box_node* gpu_children;
        hipMalloc(&gpu_children, size_of_children);
        hipMemcpy(gpu_children, cpu_node->children, size_of_children, hipMemcpyHostToDevice);
        cpu_node->children = gpu_children;

        int total_size = size_of_children + sizeof(box_node);
        box_node* gpu_node;
        hipMalloc(&gpu_node, total_size);
        hipMemcpy(gpu_node, cpu_node, total_size, hipMemcpyHostToDevice);
        return gpu_node;
    }


    int size_of_children = sizeof(box_node) * cpu_node->num_children;

    box_node* gpu_children;
    hipMalloc(&gpu_children, size_of_children);
    hipMemcpy(gpu_children, cpu_node->children, size_of_children, hipMemcpyHostToDevice);

    cpu_node->children = gpu_children;

    int total_size = size_of_children + sizeof(box_node);
    box_node* gpu_node;
    hipMalloc(&gpu_node, total_size);
    hipMemcpy(gpu_node, cpu_node, total_size, hipMemcpyHostToDevice);

    return gpu_node;
}