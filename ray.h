#ifndef RAYH
#define RAYH
#include "vec3.h"

class ray {
public:
    ray() {}
    ray(const vec3& a, const vec3& b, double time = 0.0) { 
        O = a; D = b; tm = time;
    }

    vec3 origin() const { return O; }
    vec3 direction() const { return D; }
    double time() const { return tm; }
    
    vec3 point_at_parameter(float t) const {
        return O + t * D;
    }

private:
    vec3 O;
    vec3 D;
    double tm;
};

#endif