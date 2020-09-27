#include <iostream>
#include <fstream>
#include "ray.h"
#include "hitable.h"
#include "camera.h"
#include "material.h"

vec3 color(const ray& r, hitable *world, int depth){
  hit_record rec;
  
  //Limits number of ray bounces to 50
  if (depth > 50)
    return vec3(0,0,0);

  if (world->hit(r, 0.0001, MAXFLOAT, rec)){
    ray scattered;
    vec3 attenuation;
    if rec.mat_ptr->scatter(r, rec, attenuation, scattered)
      return attenuation*color(scattered, world, depth+1);
    return vec3(0,0,0);
  }

  //Background
  vec3 unit_direction = unit_vector(r.direction());
  float t = 0.5*(unit_direction.y() + 1.0);
  return (1.0-t)*vec3(1.,1.,1.) + t*vec3(0.5,0.7,1.0);
}

int main()
{
  std::ofstream f;
  f.open("output.ppm", std::ios::out);

  //define rendering limits/properties
  int nx = 200;
  int ny = 100;
  int ns = 100;
  camera cam;

  //define worldA
  hitable *listA[2];
  listA[0] = new sphere(vec3(0,0,-1),0.5, new lambertian(vec3(0.8,0.8,0.8)));
  listA[1] = new sphere(vec3(0,-100.5,-1),100, new lambertian(vec3(0.3,0.3,0.3)));
  hitable *worldA = new hitable_list(listA,2);

  //define worldB
  hitable *listB[4];
  listB[0] = new sphere(vec3(0,0,-1),0.5, new lambertian(vec3(0.8,0.3,0.3)));
  listB[1] = new sphere(vec3(0,-100.5,-1),100, new lambertian(vec3(0.8,0.8,0)));
  listB[2] = new sphere(vec3(1,0,-1),0.5, new metal(vec3(0.8,0.6,0.2)));
  listB[3] = new sphere(vec3(-1,0,-1),0.5, new metal(vec3(0.8,0.8,0.8)));
  hitable *worldB = new hitable_list(listB,4);

  std::cout << "Started rendering..." << std::endl;
  f << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j=ny-1; j>=0; j--)
  {
    std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i=0; i<nx; i++)
    {
      vec3 col(0,0,0);
      for (int s=0; s<ns; s++)
      {
        float u = float(i + drand48())/nx;
        float v = float(j + drand48())/ny;
        ray r = cam.get_ray(u,v);
        col += color(r, worldB, 0);
      }
      col /= float(ns);
      col[0] = sqrt(col[0]);
      col[1] = sqrt(col[1]);
      col[2] = sqrt(col[2]);
      col *= 255.99;
      f << int(col.e[0]) << " " << int(col.e[1]) << " " << int(col.e[2]) << "\n";
    }
  }
  
  f.close();
  std::cout << "Rendering done!" << std::endl;
}
