#pragma once
#include <iostream>
#include <random>
#include <cmath>
#include <curand_kernel.h>

using std::sqrt;

class vec3 {
  public:
  __host__ __device__ vec3(): e{0,0,0} {}
  __host__ __device__ vec3(float e0, float e1, float e2) : e{e0,e1,e2} {}
  __host__ __device__ float x() const {return e[0]; }
  __host__ __device__ float y() const {return e[1]; }
  __host__ __device__ float z() const {return e[2]; }
  __host__ __device__ float r() const {return e[0]; }
  __host__ __device__ float g() const {return e[1]; }
  __host__ __device__ float b() const {return e[2]; }

  __host__ __device__ const vec3& operator+() const {return *this; }
  __host__ __device__ vec3 operator-() const {return vec3(-e[0],-e[1],-e[2]); }
  __host__ __device__ float operator[](int i) const {return e[i]; }
  __host__ __device__ float& operator[](int i) {return e[i]; }

  __host__ __device__ vec3& operator+=(const vec3 &v2);
  __host__ __device__ vec3& operator-=(const vec3 &v2);
  __host__ __device__ vec3& operator*=(const vec3 &v2);
  __host__ __device__ vec3& operator/=(const vec3 &v2);
  __host__ __device__ vec3& operator*=(const float t);
  __host__ __device__ vec3& operator/=(const float t);

  __host__ __device__ float length() const { return sqrt(squared_length()); }
  __host__ __device__ float squared_length() const { return e[0]*e[0]+e[1]*e[1]+e[2]*e[2]; }
  __host__ __device__ void make_unit_vector();

  float e[3];
};

inline std::istream& operator>>(std::istream &is, vec3 &t){
  is >> t.e[0] >> t.e[1] >> t.e[2];
  return is;
}

inline std::ostream& operator<<(std::ostream &os, vec3 &t){
  os << t.e[0] << " " << t.e[1] << " " << t.e[2];
  return os;
}

__host__ __device__ inline void vec3::make_unit_vector(){
  float k = 1.0f/length();
  e[0] *= k; e[1]*= k; e[2]*= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]/v2.e[0], v1.e[1]/v2.e[1], v1.e[2]/v2.e[2]);
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const float t){
  return vec3(v1.e[0]*t, v1.e[1]*t, v1.e[2]*t);
}

__host__ __device__ inline vec3 operator*(const float t, const vec3 &v1){
  return vec3(v1.e[0]*t, v1.e[1]*t, v1.e[2]*t);
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const float t){
  return vec3(v1.e[0]/t, v1.e[1]/t, v1.e[2]/t);
}

__host__ __device__ inline float dot(const vec3 &v1, const vec3 &v2){
  return v1.e[0]*v2.e[0]+v1.e[1]*v2.e[1]+v1.e[2]*v2.e[2];
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1],
              -v1.e[0]*v2.e[2] + v1.e[2]*v2.e[0],
              v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]);
}

__host__ __device__ inline vec3& vec3::operator+=(const vec3& v){
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3& v){
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3& v){
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v){
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const float t){
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const float t){
  float k = 1.0f/t;
  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v){
  return v/v.length();
}

__host__ __device__ inline float clamp(float x, float min, float max){
  if (x<min) return min;
  if (x>max) return max;
  return x;
}

__device__ inline float random_float(curandState* s, float min=0, float max=1){
  return min + (max-min)*curand_uniform(s);
}

__device__ inline vec3 random_vec3(curandState* s, float min=0, float max=1){
    return vec3(random_float(s,min,max),random_float(s,min,max),random_float(s,min,max));
}

__device__ vec3 random_in_unit_sphere(curandState* s){
  auto a = random_float(s, 0, 2*M_PI);
  auto z = random_float(s, -1,1);
  auto r = sqrt(1-z*z);
  return vec3(r*cos(a), r*sin(a), z);
}
