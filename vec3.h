#pragma once
#include <iostream>
#include <random>
#include <cmath>

using std::sqrt;

class vec3 {
  public:
  vec3(): e{0,0,0} {}
  vec3(float e0, float e1, float e2) : e{e0,e1,e2} {}
  float x() const {return e[0]; }
  float y() const {return e[1]; }
  float z() const {return e[2]; }
  float r() const {return e[0]; }
  float g() const {return e[1]; }
  float b() const {return e[2]; }

  const vec3& operator+() const {return *this; }
  vec3 operator-() const {return vec3(-e[0],-e[1],-e[2]); }
  float operator[](int i) const {return e[i]; }
  float& operator[](int i) {return e[i]; }

  vec3& operator+=(const vec3 &v2);
  vec3& operator-=(const vec3 &v2);
  vec3& operator*=(const vec3 &v2);
  vec3& operator/=(const vec3 &v2);
  vec3& operator*=(const float t);
  vec3& operator/=(const float t);

  float length() const { return sqrt(squared_length()); }
  float squared_length() const { return e[0]*e[0]+e[1]*e[1]+e[2]*e[2]; }
  void make_unit_vector();

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

inline void vec3::make_unit_vector(){
  float k = 1.0/length();
  e[0] *= k; e[1]*= k; e[2]*= k;
}

inline vec3 operator+(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]+v2.e[0], v1.e[1]+v2.e[1], v1.e[2]+v2.e[2]);
}

inline vec3 operator-(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]-v2.e[0], v1.e[1]-v2.e[1], v1.e[2]-v2.e[2]);
}

inline vec3 operator*(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]*v2.e[0], v1.e[1]*v2.e[1], v1.e[2]*v2.e[2]);
}

inline vec3 operator/(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[0]/v2.e[0], v1.e[1]/v2.e[1], v1.e[2]/v2.e[2]);
}

inline vec3 operator*(const vec3 &v1, const float t){
  return vec3(v1.e[0]*t, v1.e[1]*t, v1.e[2]*t);
}

inline vec3 operator*(const float t, const vec3 &v1){
  return vec3(v1.e[0]*t, v1.e[1]*t, v1.e[2]*t);
}

inline vec3 operator/(const vec3 &v1, const float t){
  return vec3(v1.e[0]/t, v1.e[1]/t, v1.e[2]/t);
}

inline float dot(const vec3 &v1, const vec3 &v2){
  return v1.e[0]*v2.e[0]+v1.e[1]*v2.e[1]+v1.e[2]*v2.e[2];
}

inline vec3 cross(const vec3 &v1, const vec3 &v2){
  return vec3(v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1],
              -v1.e[0]*v2.e[2] + v1.e[2]*v2.e[0],
              v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]);
}

inline vec3& vec3::operator+=(const vec3& v){
  e[0] += v.e[0];
  e[1] += v.e[1];
  e[2] += v.e[2];
  return *this;
}

inline vec3& vec3::operator*=(const vec3& v){
  e[0] *= v.e[0];
  e[1] *= v.e[1];
  e[2] *= v.e[2];
  return *this;
}

inline vec3& vec3::operator/=(const vec3& v){
  e[0] /= v.e[0];
  e[1] /= v.e[1];
  e[2] /= v.e[2];
  return *this;
}

inline vec3& vec3::operator-=(const vec3& v){
  e[0] -= v.e[0];
  e[1] -= v.e[1];
  e[2] -= v.e[2];
  return *this;
}

inline vec3& vec3::operator*=(const float t){
  e[0] *= t;
  e[1] *= t;
  e[2] *= t;
  return *this;
}

inline vec3& vec3::operator/=(const float t){
  float k = 1.0/t;
  e[0] *= k;
  e[1] *= k;
  e[2] *= k;
  return *this;
}

inline vec3 unit_vector(vec3 v){
  return v/v.length();
}

inline float random_float(float min=0, float max=1){
    static std::uniform_real_distribution<float> distribution(min, max);
    static std::mt19937 generator;
    return distribution(generator);
}

inline float clamp(float x, float min, float max){
  if (x<min) return min;
  if (x>max) return max;
  return x;
}

inline vec3 random_vec3(float min=0, float max=1){
    return vec3(random_float(min,max),random_float(min,max),random_float(min,max));
}

vec3 random_in_unit_sphere(){
  while(true){
    auto p = random_vec3(-1,1);
    if (p.squared_length() >= 1) continue;
    return p;
  }
}

