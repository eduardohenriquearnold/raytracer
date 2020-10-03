#include <iostream>
#include <fstream>
#include "ray.h"
#include "hitable.h"
#include "camera.h"
#include "material.h"

hitable_list randomScene(){
  hitable_list world;

  auto ground_material = make_shared<lambertian>(vec3(0.5, 0.5, 0.5));
	world.add(make_shared<sphere>(vec3(0,-1000,0), 1000, ground_material));

  for (int a=-11; a<11; a++)
    for (int b=-11; b<11; b++) {
      float choose_mat = random_float();
      vec3 center(a+random_float(0,0.9), 0.2, b+random_float(0,0.9));

      if ((center-vec3(4,0.2,0)).length() < 0.9)
        continue;

      shared_ptr<material> mat;
      if (choose_mat < 0.8)
        mat = make_shared<lambertian>(random_vec3());
      else if (choose_mat < 0.95)
        mat = make_shared<metal>(random_vec3(0,0.5));
      else
        mat = make_shared<dielectric>(1.5);

      world.add(make_shared<sphere>(center, 0.2, mat));
    }

  auto material1 = make_shared<dielectric>(1.5);
  world.add(make_shared<sphere>(vec3(0, 1, 0), 1.0, material1));

  auto material2 = make_shared<lambertian>(vec3(0.4, 0.2, 0.1));
  world.add(make_shared<sphere>(vec3(-4, 1, 0), 1.0, material2));

  auto material3 = make_shared<metal>(vec3(0.7, 0.6, 0.5));
  world.add(make_shared<sphere>(vec3(4, 1, 0), 1.0, material3));
  
  return world;
}

vec3 color(const ray& r, const hitable& world, int depth){
  hit_record rec;
  
  //Limits number of ray bounces to 50
  if (depth > 50)
    return vec3(0,0,0);

  if (world.hit(r, 0.0001, MAXFLOAT, rec)){
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
  const int nx = 600;
  const int ny = 400;
  const int ns = 100;
	camera cam(vec3(13,2,3), vec3(0,0,0), vec3(0,1,0), 20, float(nx)/float(ny));
	// camera cam(vec3(0,0,0), vec3(0,0,-1), vec3(0,1,0), 90, float(nx)/ny);

  //define worldA
  hitable_list worldA;
  worldA.add(make_shared<sphere>(vec3(0,0,-1),0.5, make_shared<lambertian>(vec3(0.8,0.8,0.8))));
  worldA.add(make_shared<sphere>(vec3(0,-100.5,-1),100, make_shared<lambertian>(vec3(0.3,0.3,0.3))));

  //define worldB
  hitable_list worldB;
  worldB.add(make_shared<sphere>(vec3(0,0,-1),0.5, make_shared<lambertian>(vec3(0.8,0.3,0.3))));
  worldB.add(make_shared<sphere>(vec3(0,-100.5,-1),100, make_shared<lambertian>(vec3(0.8,0.8,0))));
  worldB.add(make_shared<sphere>(vec3(1,0,-1),0.5, make_shared<metal>(vec3(0.8,0.6,0.2))));
  worldB.add(make_shared<sphere>(vec3(-1,0,-1),0.5, make_shared<dielectric>(1.5)));

  //random scene
  hitable_list worldC = randomScene();

  //render world
  hitable_list world = worldC;
  std::cout << "Started rendering..." << std::endl;
  f << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j=ny-1; j>=0; j--)
  {
    std::cout << "\rScanlines remaining: " << j << ' ' << std::flush;
    for (int i=0; i<nx; i++)
    {
      vec3 col(0,0,0);
			#pragma omp parallel for
      for (int s=0; s<ns; s++)
      {
        float u = (i + random_float())/nx;
        float v = (j + random_float())/ny;
        ray r = cam.get_ray(u,v);
        col += color(r, world, 0);
      }
      col /= ns;
      col[0] = clamp(sqrt(col[0]), 0, 0.999);
      col[1] = clamp(sqrt(col[1]), 0, 0.999);
      col[2] = clamp(sqrt(col[2]), 0, 0.999);
      col *= 256;
      f << int(col.e[0]) << " " << int(col.e[1]) << " " << int(col.e[2]) << "\n";
    }
  }
  
  f.close();
  std::cout << "Rendering done!" << std::endl;
}
