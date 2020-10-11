#pragma once
#include "vec3.h"
#include "ray.h"
#include "hitable.h"

class material {
  public: 
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const = 0;
};

class lambertian: public material{
  public:
    __device__ lambertian(const vec3& a) : albedo(a) {}
    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {      
      vec3 target = rec.p + rec.normal + random_in_unit_sphere();
      scattered = ray(rec.p, target-rec.p);
      attenuation = albedo;
      return true;
    }

    vec3 albedo;
};

class metal: public material{
  public:
    __device__ metal(const vec3& a) : albedo(a) {}

    __device__ vec3 reflect(const vec3& v, const vec3& n) const {
      return v-2*dot(v,n)*n;
    }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {      
      vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
      scattered = ray(rec.p, reflected);
      attenuation = albedo;
      return dot(scattered.direction(), rec.normal) > 0;
    }

    vec3 albedo;
};

class dielectric : public material{
  public:
    __device__ dielectric(float ri) : ref_idx(ri){}

    __device__ vec3 reflect(const vec3& v, const vec3& n) const {
      return v-2*dot(v,n)*n;
    }

    __device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) const {
      vec3 uv = unit_vector(v);
      float dt = dot(uv, n);
      float discriminant = 1 - ni_over_nt*ni_over_nt*(1-dt*dt);
      if (discriminant >0) {
        refracted = ni_over_nt*(uv-n*dt) - n*sqrt(discriminant);
        return true;
      }
      else
        return false;
    }

    __device__ float schlick(float cosine) const{
      float r0 = (1-ref_idx)/(1+ref_idx);
      r0 = r0*r0;
      return r0 + (1-r0)*pow(1-cosine, 5);
    }

    __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered) const {      
      vec3 outward_normal;
      vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
      float ni_over_nt;
      attenuation = vec3(1,1,1);
      vec3 refracted;
      float reflect_prob;
      float cosine;

      if (dot(r_in.direction(), rec.normal)>0) {
        outward_normal = -rec.normal;
        ni_over_nt = ref_idx;
        cosine = ref_idx * dot(r_in.direction(), rec.normal) / r_in.direction().length();
      }
      else {
        outward_normal = rec.normal;
        ni_over_nt = 1.0/ref_idx;
        cosine = -dot(r_in.direction(), rec.normal) / r_in.direction().length();
      }

      if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted)) 
        reflect_prob = schlick(cosine);
      else 
        reflect_prob = 1;

      if (drand48() < reflect_prob)
        scattered = ray(rec.p, reflected);
      else
        scattered = ray(rec.p, refracted);

      return true;
    }

    float ref_idx;

};
