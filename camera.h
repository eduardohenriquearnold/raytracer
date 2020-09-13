#pragma once
#include "ray.h"

class camera{
  public:
    camera() : origin(0,0,0), horizontal(4,0,0), vertical(0,2,0), lower_left_corner(-2,-1,-1) {}
    ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin);}

    vec3 origin;
    vec3 horizontal;
    vec3 vertical;
    vec3 lower_left_corner;
};
