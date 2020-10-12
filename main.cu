#include <iostream>
#include <fstream>
#include "ray.h"
#include "camera.h"
#include "hitable.h"
#include "material.h"

__device__ vec3 color(const ray& r) {
      vec3 unit_direction = unit_vector(r.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, const int nx, const int ny, camera* c, curandState *crs) {
  int i = blockIdx.x * blockDim.x +threadIdx.x;
  int j = blockIdx.y * blockDim.y +threadIdx.y;
  if (i>=nx || j>=ny)
    return;
  int pixelIdx = i + j*nx;
  float u = float(i) / float(nx);
  float v = float(j) / float(ny);
  ray r = c->get_ray(u,v);
  vec3 col = color(r);
  col[0] = clamp(sqrt(col[0]), 0, 0.999);
  col[1] = clamp(sqrt(col[1]), 0, 0.999);
  col[2] = clamp(sqrt(col[2]), 0, 0.999);
  col *= 256;
  fb[pixelIdx] = col;
}

__global__ void init_random_states(curandState *s, const int nx, const int ny) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= nx) || (j >= ny)) return;
  int pixel_index = i + j*nx;

  //Each thread gets different seed and same sequence number (more efficient than using same seed with different sequence number, according to NVIDIA docs)
  curand_init(1984+pixel_index, 0, 0, &s[pixel_index]);    
}

int main()
{
  std::ofstream f;
  f.open("output.ppm", std::ios::out);

  //define rendering limits/properties
  const int nx = 600;
  const int ny = 400;
  const int ns = 100;

  //define thread numbers and block dimensions
  int tx = 8;
  int ty = 8;
  dim3 blocks(nx/tx+1, ny/ty+1);
  dim3 threads(tx,ty);

  //allocate device memory for image
  vec3 *fb;
  cudaMallocManaged(&fb, sizeof(vec3)*nx*ny);

  //create random states
  curandState *crs;
  cudaMalloc(&crs, nx*ny*sizeof(curandState));
  init_random_states<<<blocks,threads>>>(crs, nx, ny);

  //create camera (in unified memory, thanks to Managed class!)
  // camera *cam = new camera(vec3(13,2,3), vec3(0,0,0), vec3(0,1,0), 20, float(nx)/float(ny));
  camera *cam = new camera(vec3(0,0,0), vec3(0,0,-1), vec3(0,1,0), 90, float(nx)/ny);
  
  //create world (objects and materials) in device
  hitable_list *world = new hitable_list(10);
  world->add(new sphere(vec3(0,-100.5,-1), 100.f, new lambertian(vec3(0.3,0.3,0.3))));
  world->add(new sphere(vec3(0,0,-1), 0.5f, new lambertian(vec3(0.3,0.5,0.3)) ));

  //render world
  std::cout << "Started rendering..." << std::endl;
  render<<<blocks, threads>>>(fb, nx, ny, cam, crs);
  cudaDeviceSynchronize();
  std::cout << "Finished rendering." << std::endl;

  //save image
  f << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j=ny-1; j>=0; j--)
    for (int i=0; i<nx; i++)
    {
      vec3 col = fb[i+j*nx];
      f << int(col.e[0]) << " " << int(col.e[1]) << " " << int(col.e[2]) << "\n";
    }
  f.close();
  std::cout << "Saved image." << std::endl;

  //free memory
  cudaFree(fb);
  delete cam;
  delete world;
}
