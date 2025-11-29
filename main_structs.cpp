// A library file with all of the required main structs and their constructors, plus some methods to use on them

// // These are the main, arbitrary structs. Structs designed for a more niche circumstance (such as a collision struct that holds a 
// collision point vector, a collision triangle, and a collision distance) will be found in the specific_structs.cpp file

// 3D vector with x-, y-, and z-values
struct vector {
    double* x = new double[1];
    double* y = new double[1];
    double* z = new double[1];

    __device__ __host__ vector() {}              // An empty constructor, only used for when allocating memory for a vector to take up in the future (i.e. 
                                        // through "new vector[1]") -- same goes for all other empty constructors following
    
    __device__ __host__ vector(double _x, double _y, double _z) {
        *x = _x;
        *y = _y;
        *z = _z;
    }

    __device__ __host__ ~vector() {
        delete x, y, z;
    }

    // Makes a deep copy of/clones this vector (where new vector values are totally separate from the this vector's values)
    __device__ __host__ vector* clone() {
        return new vector(*x, *y, *z);
    }

    // Tranforms the given vector by the given matrix
    __device__ __host__ void transform(double* matrix) {
        double result[] = {(matrix[0] + matrix[1] + matrix[2]) * *x, 
                           (matrix[3] + matrix[4] + matrix[5]) * *y,
                           (matrix[6] + matrix[7] + matrix[8]) * *z};
        *x = result[0];
        *y = result[1];
        *z = result[2];
    }


    // Subtracts the given vector from this vector
    __device__ __host__ void sub(vector* v) {
        *x -= *v->x;
        *y -= *v->y;
        *z -= *v->z;
    }

    // Adds the given vector to this vector
    __device__ __host__ void add(vector* v) {
        *x += *v->x;
        *y += *v->y;
        *z += *v->z;
    }

    // 3D vector rotation methods that rotate this around the given vector center by the given radians, on the respective axis
    __device__ __host__ void rotate_x(vector* center, double radians) {
        double sine = sin(radians);
        double cosine = cos(radians);
        
        double transformation_matrix[] = {
            1, 0, 0,
            0, cosine, -sine,
            0, sine, cosine
        };
        
        sub(center);
        transform(transformation_matrix);
        add(center);
    }

    __device__ __host__ void rotate_y(vector* center, double radians) {
        double sine = sin(radians);
        double cosine = cos(radians);
        
        double transformation_matrix[] = {
            cosine, 0, sine,
            0, 1, 0,
            -sine, 0, cosine
        };
        
        sub(center);
        transform(transformation_matrix);
        add(center);
    }

    __device__ __host__ void rotate_z(vector* center, double radians) {
        double sine = sin(radians);
        double cosine = cos(radians);
        
        double transformation_matrix[] = {
            cosine, -sine, 0,
            sine, cosine, 0,
            0, 0, 1
        };
        
        sub(center);
        transform(transformation_matrix);
        add(center);
    }

    // Returns the magnitude (or length) of this vector
    __device__ __host__ double magnitude() {
        double sum = (*x * *x) + (*y * *y) + (*z * *z);
        return sqrt(sum);
    }

    // Shortens this vector to a length of 1
    __device__ __host__ void normalize() {
        double mag = magnitude();
        *x /= mag;
        *y /= mag;
        *z /= mag;
    }

    // Returns the dot product of the given vector and this vector
    __device__ __host__ double dot(vector* v) {
        return (*x * *v->x) + (*y * *v->y) + (*z * *v->z);
    }

    // Returns the cross product of the this vector and the given vector
    __device__ __host__ vector* cross(vector* v) {
        vector* result = new vector((*y * *v->z) - (*z * *v->y),
                                    (*z * *v->x) - (*x * *v->z),
                                    (*x * *v->y) - (*y * *v->x));
        return result;
    }
};

// RGB color with values between 0-1 (no alpha)
struct color {
    double* r = new double[1];
    double* g = new double[1];
    double* b = new double[1];

    __device__ __host__ color() {}

    __device__ __host__ color(double _r, double _g, double _b) {
        *r = _r;
        *g = _g;
        *b = _b;
    }

    __device__ __host__ ~color() {
        delete r, g, b;
    }
};

// A template for a material with different parameters that control how the material interacts with light (used in calculating BRDFs)
struct material {
    color* material_color = new color[1];
    double* diffusion = new double[1];
    double* reflection = new double[1];
    double* refraction = new double[1];

    __device__ __host__ material() {}

    __device__ __host__ material(color* _material_color, double _diffusion, double _reflection, double _refraction) {
        material_color = _material_color;
        *diffusion = _diffusion;
        *reflection = _reflection;
        *refraction = _refraction;
    }

    __device__ __host__ ~material() {
        delete material_color, diffusion, reflection, refraction;
    }
};

// A 3D plane with components a, b, c, d, expressed by equation ax + by + cz + d = 0
struct plane {
    vector* normal = new vector[1];                     // vector to store the components of the plane's normal as (a, b, c)
    double* d = new double[1];

    __device__ __host__ plane() {}

    __device__ __host__ plane(vector* _normal, double _d) {
        normal = _normal;
        *d = _d;
    }

    __device__ __host__ ~plane() {
        delete normal, d;
    }
};

// A 3D Ray that starts from the 3D point origin and points in the direction given by the direction vector
struct ray {
    vector* origin = new vector[1];
    vector* direction = new vector[1];

    __device__ __host__ ray() {}

    __device__ __host__ ray(vector* _origin, vector* _direction) {
        origin = _origin;
        direction = _direction;
    }

    __device__ __host__ ~ray() {
        delete origin, direction;
    }
};

// A 3D triangle defined by the 3 vectors a, b, and c, with the given material and plane
struct triangle {
    plane* surface_plane = new plane[1];               // the 3D plane that the triangle sits on
    material* surface_material = new material[1];         // the material that the triangle is "made out of," defining how light rays should interact with the triangle
    vector* a = new vector[1];
    vector* b = new vector[1];
    vector* c = new vector[1];

    __device__ __host__ triangle() {}

    __device__ __host__ triangle(plane* _surface_plane, material* _surface_material, vector* _a, vector* _b, vector* _c) {
        surface_plane = _surface_plane;
        surface_material = _surface_material;
        a = _a;
        b = _b;
        c = _c;
    }

    // Alternate triangle constructor that doesn't require a plane
    __device__ __host__ triangle(material* _surface_material, vector* _a, vector* _b, vector* _c) {
        // Setting the struct's members
        surface_material = _surface_material;
        a = _a;
        b = _b;
        c = _c;


        // Here we are taking the cross product of the vectors that make up two of the legs of the triangle to find the normal of the plane that the 
        // triangle sits on, because both of them are by definition situated on the same plane as the triangle, to find a vector that is parallel to both, 
        // which is equivalent to the normal of the plane
        vector* ab = a->clone();
        vector* bc = b->clone();
        ab->sub(b);
        bc->sub(c);

        vector* plane_normal = ab->cross(bc);
        plane_normal->normalize();                                                           // Normalizing to help with calculations
        
        // Now we need to calculate the shift of the plane, aka d in the plane's equation
        // We do this by substituting in the coordinates for a known point that lies on the plane. What points do we know? Well, any of the 3 vertices of 
        // the triangle will work, because they define the plane of the triangle so they by definition lie on it
        double* plane_a = plane_normal->x;
        double* plane_b = plane_normal->y;
        double* plane_c = plane_normal->z;

        double* x0 = a->x;
        double* y0 = a->y;
        double* z0 = a->z;

        double d = -((*plane_a * *x0) + (*plane_b * *y0) + (*plane_c * *z0));               // We are making the shift negative here because of how the 
                                                                                            // plane equation is arranged (in this code, at least): ax + 
                                                                                            // by + cz + d = 0, where we are plugging in known values for 
                                                                                            // ax, by, and cz, and solving for d
        surface_plane = new plane(plane_normal, d);

        // Deleting local variables to free memory
        delete plane_a, plane_b, plane_c, x0, y0, z0, ab, bc;
    }

    __device__ __host__ ~triangle() {
        delete surface_plane, surface_material, a, b, c;
    }
};

// A container to hold the height and width of an image (or anything else with height and width)
struct dimensions {
    int* width = new int[1];
    int* height = new int[1];

    __device__ __host__ dimensions() {}

    __device__ __host__ dimensions(int _width, int _height) {
        *width = _width;
        *height = _height;
    }

    __device__ __host__ ~dimensions() {
        delete width, height;
    }
};

// 3D camera, defines where the camera rays originate and in which direction they radiate, to control where the viewport is looking
struct camera {
    vector* origin = new vector[1];     // The 3D point where all camera rays originate from
    vector* rotation = new vector[1];   // The direction where camera rays radiate from the origin, with components (x_rotation, y_rotation, 
    // z_rotation)
    double* fov_scale = new double[1];  // The field-of-view parameters, expressed in radians, that define how far left/right or up/down the camera 
    // can see

    __device__ __host__ camera() {}

    __device__ __host__ camera(vector* _origin, vector* _rotation, double _fov_scale) {
        origin = _origin;
        rotation = _rotation;
        *fov_scale = _fov_scale;
    }

    __device__ __host__ ~camera() {
        delete origin, rotation, fov_scale;
    }
};

// 3D point-source light with color, position, and intensity
struct light {
    vector* position = new vector[1];
    color* rgb = new color[1];
    double* intensity = new double[1];

    __device__ __host__ light() {}

    __device__ __host__ light(vector* _position, color* _rgb, double _intensity) {
        position = _position;
        rgb = _rgb;
        *intensity = _intensity;
    }

    __device__ __host__ ~light() {
        delete position, rgb, intensity;
    }
};


// Checks if the point (i, j) is contained in the triangle defined by the points (x1, y1), (x2, y2), and (x3, y3) -- required dependency for
// ray-triangle intersection method below.
// Works by looking at the triangle in 2D (as if it had been orthogonally projected to a 2D plane) and seeing if the given point lies on the same side
// of all 3 of the triangles sides, using the dot products of the vectors that make up the triangle (rotating the point 90 degrees before taking the
// dot product, so that we get which side the point is on as the sign of the dot product, instead of whether the vector that encodes the point is
// aligned with the triangle leg vectors that we are checking).
// Basic intuition is to imagine walking clockwise along the outside of the triangle, and if the point being checked stays on your righthand side the
// entire time you are walking, then it must be inside the triangle and not outside
__device__ bool contains(double i, double j, double x1, double y1, double x2, double y2, double x3, double y3) {
    double dotAB = -(i - y1) * (x2 - x1) + (j - x1) * (y2 - y1);
    double dotBC = -(i - y2) * (x3 - x2) + (j - x2) * (y3 - y2);
    double dotCA = -(i - y3) * (x1 - x3) + (j - x3) * (y1 - y3);
    
    bool allPos = dotAB >= 0 && dotBC >= 0 && dotCA >= 0;
    bool allNeg = dotAB <= 0 && dotBC <= 0 && dotCA <= 0;

    return allPos || allNeg;
}

// Same method as above using pointers to save memory on unnecessary variable declarations (hopefully -- I'm not too familiar with C++ still so I'm not
// sure if this is actually helping or hurting or if the compiler figures it all out no matter what and it's really the same either way)
__device__ bool contains(double* i, double* j, double* x1, double* y1, double* x2, double* y2, double* x3, double* y3) {
    double dotAB = -(*i - *y1) * (*x2 - *x1) + (*j - *x1) * (*y2 - *y1);
    double dotBC = -(*i - *y2) * (*x3 - *x2) + (*j - *x2) * (*y3 - *y2);
    double dotCA = -(*i - *y3) * (*x1 - *x3) + (*j - *x3) * (*y1 - *y3);
    
    bool allPos = dotAB >= 0 && dotBC >= 0 && dotCA >= 0;
    bool allNeg = dotAB <= 0 && dotBC <= 0 && dotCA <= 0;

    return allPos || allNeg;
}


// Returns the t-value (or distance) where the given ray intersects the given plane
// If an intersection point exists, has_intersection will be set to true, otherwise (i.e. if plane is behind ray or if plane and ray are parallel),
// has_intersection will be set to false (if there is no intersection, the function will also return 0)
// To find the intersection point, we need to plug in the ray components (parameterized using the parameter t) into the plane equation and solve.
// For the ray components to be "paramaterized," it means that the origin and direction x-, y-, and z-values are expressed in terms of a constant "t."
// Changing this "t" constant gives x-, y-, and z-values corresponding to the point along the ray that is equal to origin + t * direction.
// To express x-, y-, and z-values in terms of t: x = x0 + xt, y = y0 + yt, and z = z0 + zt, where (x0, y0, z0) is the origin and (xt, yt, zt) is the
// direction of the ray. Substituting these into the plane's equation ax + by + cz + d = 0 and solving gives us the intersection point.
__device__ double* ray_plane_intersection_t(ray* r, plane* p, bool* has_intersection) {
    if (r->direction->dot(p->normal) == 0) {
        *has_intersection = false;
        return 0;
    } else {
        *has_intersection = true;
    }
    double* a = p->normal->x;
    double* b = p->normal->y;
    double* c = p->normal->z;
    double* d = p->d;

    double* x0 = r->origin->x;
    double* y0 = r->origin->y;
    double* z0 = r->origin->z;
    
    double* xt = r->direction->x;
    double* yt = r->direction->y;
    double* zt = r->direction->z;

    
    double left = (*a * *xt) + (*b * *yt) + (*c * *zt);              // The total t-values added up in the ray-plane equation being solved -- this 
                                                                        // is negative because we are subtracting the values from the left side of the 
                                                                        // equation to the right side of the equation
    double right = -((*a * *x0) + (*b * *y0) + (*c * *z0) + *d);           // The total constants added up in the ray-plane equation being solved
                                                                        
                                                                        

    double* result = new double[1];
    result[0] = right / left;
    return result;                                                     // After the last step, the equation is something like c = kt, where c and k are some given constants, and 
                                                                        // we need to isolate t so we divide both sides by k to get t = c / k
}


// Gets the 3D point from a ray at a given t value (where the ray equation is O + vt, where O is the ray origin and v is the ray's direction -- this
// function just plugs in a given t and returns the 3D point that results from the equation)
__device__ vector* get_point_from_t(ray* r, double* t) {
    vector* origin = r->origin;
    vector* direction = r->direction;
    vector* result = new vector(*origin->x + (*direction->x * *t),
                                *origin->y + (*direction->y * *t),
                                *origin->z + (*direction->z * *t));
    return result;
}


// In the future, will use the Möller–Trumbore ray-triangle intersection algorithm, actually much simpler than I had previously thought.
// https://en.wikipedia.org/wiki/Möller–Trumbore_intersection_algorithm
// Basically, instead of using a precalculated plane normal for the plane the given triangle sits on, we express the plane in terms of two of the 
// vectors that make up the legs of the triangle.
// By scaling these two leg vectors by the barycentric coordinates u and v of the triangle, we can essentially use them as basis vectors for the plane,
// and describe any point on the plane with just u and v.
// My explanation here is very lacking as I have just started understanding the Möller–Trumbore algorithm, so if you want more depth look at the linked
// Wikipedia page, it explains much better than I can with just code comments.
// As of now, still using same algorithm as above then just doing bounds-checking to see if the intersection point we find is inside the triangle
// With this method, however, we return the actual coordinates of the 3D collision point (if it exists) and we return the t value of the collision 
// through the t_out argument
__device__ vector* ray_triangle_intersection_t(ray* r, triangle* t, bool* has_intersection, double* t_out) {
    plane* p = t->surface_plane;
    
    if (r->direction->dot(p->normal) == 0) {
        *has_intersection = false;
        vector* result = new vector(0, 0, 0);
        *t_out = 0;
        return result;
    }

    double* a = p->normal->x;
    double* b = p->normal->y;
    double* c = p->normal->z;
    double* d = p->d;

    double* x0 = r->origin->x;
    double* y0 = r->origin->y;
    double* z0 = r->origin->z;
    
    double* xt = r->direction->x;
    double* yt = r->direction->y;
    double* zt = r->direction->z;

    
    double left = (*a * *xt) + (*b * *yt) + (*c * *zt);             // The total t-values added up in the ray-plane equation being solved -- this 
                                                                        // is negative because we are subtracting the values from the left side of the 
                                                                        // equation to the right side of the equation
    double right = -((*a * *x0) + (*b * *y0) + (*c * *z0) + *d);          // The total constants added up in the ray-plane equation being solved
                                                                        
                                                                        


    *t_out = right / left;                                             // After the last step, the equation is something like c = kt, where c and k 
                                                                        // are some given constants, and we need to isolate t so we divide both sides 
                                                                        // by k to get t = c / k
    // Now we need to find the collision point's coordinates in 3D and 
    vector* collision_point = get_point_from_t(r, t_out);

    double* x1 = t->a->x;
    double* y1 = t->a->y;
    double* x2 = t->b->x;
    double* y2 = t->b->y;
    double* x3 = t->c->x;
    double* y3 = t->c->y;
    
    double* i = collision_point->x;
    double* j = collision_point->y;

    *has_intersection = contains(i, j, x1, y1, x2, y2, x3, y3);

    if (*has_intersection) {
        return collision_point;
    } else {
        vector* result = new vector(0, 0, 0);
        *t_out = 0;
        return result;
    }
}




// Print methods for debugging
__device__ __host__ void print_vector(vector* v) {
    printf("(%f, %f, %f)", *v->x, *v->y, *v->z);
}









// triangle methods


// dimensions methods


// camera methods