#include <iostream>
#include <fstream>
#include "ray.h"
#include "hitable.h"
#include "camera.h"

vec3 color(const ray& r, hitable *world){
  hit_record rec;
  if (world->hit(r, 0, MAXFLOAT, rec))
    return 0.5*vec3(rec.normal.x()+1, rec.normal.y()+1, rec.normal.z()+1);
  else {
    //Background
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5*(unit_direction.y() + 1.0);
    return (1.0-t)*vec3(1.,1.,1.) + t*vec3(0.5,0.7,1.0);
  }

}

int main()
{
  std::ofstream f;
  f.open("output.ppm", std::ios::out);

  //define rendering limits/properties
  int nx = 600;
  int ny = 300;
  int ns = 100;
  camera cam;

  //define world
  hitable *list[2];
  list[0] = new sphere(vec3(0,0,-1),0.5);
  list[1] = new sphere(vec3(0,-100.5,-1),100);
  hitable *world = new hitable_list(list,2);

  std::cout << "Started rendering..." << std::endl;
  f << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j=ny-1; j>=0; j--)
    for (int i=0; i<nx; i++)
    {
      vec3 col(0,0,0);
      for (int s=0; s<ns; s++)
      {
        float u = float(i + drand48())/nx;
        float v = float(j + drand48())/ny;
        ray r = cam.get_ray(u,v);
        col += color(r, world);
      }
      col /= float(ns);
      col *= 255.99;
      f << int(col.e[0]) << " " << int(col.e[1]) << " " << int(col.e[2]) << "\n";
    }
  
  f.close();
  std::cout << "Rendering done!" << std::endl;
}
