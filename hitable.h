#pragma once
#include "ray.h"

class material;

struct hit_record {
  float t;
  vec3 p;
  vec3 normal;
  material *mat_ptr;
};

class hitable{
  public:
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

class hitable_list : public hitable {
  public:
    hitable_list() {}
    hitable_list(hitable **l, int n){list=l; list_size=n;}
    virtual bool hit(const ray& r, float tmin, float t_max, hit_record& rec) const;

    hitable **list;
    int list_size;
};

bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& rec) const{
  hit_record temp_rec;
  bool hit_anything = false;
  double closest_so_far = t_max;
  for (int i=0; i<list_size; i++)
    if (list[i]->hit(r, t_min, closest_so_far, temp_rec))
    {
      hit_anything = true;
      closest_so_far = temp_rec.t;
      rec = temp_rec;
    }
  return hit_anything;
}

class sphere: public hitable{
  public:
    sphere(){}
    sphere(vec3 cen, float r, material* mat_ptr) : center(cen), radius(r), mat(mat_ptr){};
    virtual bool hit(const ray& r, float tmin, float t_max, hit_record& rec) const;

    vec3 center;
    float radius;
    material* mat;
};

bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc,oc) - radius*radius;
  float discriminant = b*b - a*c;
  if (discriminant>0) {
    float temp = (-b-sqrt(discriminant))/a;
    if (temp < t_max && temp > t_min){
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p-center)/radius;
      rec.mat_ptr = mat;
      return true;
    }
    temp = (-b+sqrt(discriminant))/a;
    if (temp < t_max && temp > t_min){
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p-center)/radius;
      rec.mat_ptr = mat;
      return true;
    }
  }
  return false;
}
