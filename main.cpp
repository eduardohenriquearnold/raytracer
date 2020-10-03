#include <iostream>
#include <fstream>
#include "ray.h"
#include "hitable.h"
#include "camera.h"
#include "material.h"

hitable* randomScene(){
  int n = 500;
  hitable **list = new hitable*[n+1];
  int i=1;
  for (int a=-11; a<11; a++)
    for (int b=-11; b<11; b++) {
      float choose_mat = drand48();
      vec3 center(a+0.9*drand48(), 0.2, b+0.9*drand48());

      if ((center-vec3(4,0.2,0)).length() < 0.9)
        continue;

      material* mat;
      vec3 albedo = random_in_unit_sphere();
      albedo = albedo*albedo;
      if (choose_mat < 0.8)
        mat = new lambertian(albedo);
      else if (choose_mat < 0.95)
        mat = new metal(0.5*albedo);
      else
        mat = new dielectric(1.5);

      list[i++] = new sphere(center, 0.2, mat);
    }

  
  list[0] = new sphere(vec3(0,-1000,0), 1000, new lambertian(vec3(0.5,0.5,0.5))); //world sphere
  list[i++] = new sphere(vec3(0,1,0), 1, new dielectric(1.5)); //big glass sphere
  list[i++] = new sphere(vec3(-4,1,0), 1, new lambertian(vec3(0.4,0.2,0.1)));
  list[i++] = new sphere(vec3(4,1,0), 1, new metal(vec3(0.7,0.6,0.5)));

  return new hitable_list(list, i);
}

vec3 color(const ray& r, hitable *world, int depth){
  hit_record rec;
  
  //Limits number of ray bounces to 50
  if (depth > 50)
    return vec3(0,0,0);

  if (world->hit(r, 0.0001, MAXFLOAT, rec)){
    ray scattered;
    vec3 attenuation;
    if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
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
  int nx = 600;
  int ny = 400;
  int ns = 10;
  camera cam(vec3(13,2,3), vec3(0,0,0), vec3(0,1,0), 90, float(nx)/float(ny));
  // camera cam(vec3(0,0,0), vec3(0,0,-1), vec3(0,1,0), 90, float(nx)/ny);

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
  listB[3] = new sphere(vec3(-1,0,-1),0.5, new dielectric(1.5));
  hitable *worldB = new hitable_list(listB,4);

  //random scene
  hitable *worldC = randomScene();

  //render world
  hitable* world = worldC;
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
        col += color(r, world, 0);
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
