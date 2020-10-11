#include <thrust/device_ptr.h>
#include <iostream>
#include <fstream>
#include "ray.h"
#include "camera.h"
  
__device__ vec3 color(const ray& r) {
      vec3 unit_direction = unit_vector(r.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
}

__global__ void render(vec3 *fb, const int nx, const int ny, camera* c) {
  int i = blockIdx.x * blockDim.x +threadIdx.x;
  int j = blockIdx.y * blockDim.y +threadIdx.y;
  if (i>=nx || j>=ny)
    return;
  int pixelIdx = i + j*nx;
  float u = float(i) / float(nx);
  float v = float(j) / float(ny);
  ray r = c->get_ray(u,v);
  fb[pixelIdx] = color(r);
}

int main()
{
  std::ofstream f;
  f.open("output.ppm", std::ios::out);

  //define rendering limits/properties
  const int nx = 600;
  const int ny = 400;
  const int ns = 100;

  //allocate device memory for image
  vec3 *fb;
  cudaMallocManaged(&fb, sizeof(vec3)*nx*ny);

  //create camera (in unified memory, thanks to Managed class!)
  // camera *cam = new camera(vec3(13,2,3), vec3(0,0,0), vec3(0,1,0), 20, float(nx)/float(ny));
  camera *cam = new camera(vec3(0,0,0), vec3(0,0,-1), vec3(0,1,0), 90, float(nx)/ny);
  
  //create world (objects and materials) in device


  //render world
  std::cout << "Started rendering..." << std::endl;
  int tx = 8;
  int ty = 8;
  dim3 blocks(nx/tx+1, ny/ty+1);
  dim3 threads(tx,ty);
  render<<<blocks, threads>>>(fb, nx, ny, cam);
  cudaDeviceSynchronize();
  std::cout << "Finished rendering." << std::endl;

  //save image
  f << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j=ny-1; j>=0; j--)
  {
    for (int i=0; i<nx; i++)
    {
      vec3 col = fb[i+j*nx];
      col[0] = clamp(sqrt(col[0]), 0, 0.999);
      col[1] = clamp(sqrt(col[1]), 0, 0.999);
      col[2] = clamp(sqrt(col[2]), 0, 0.999);
      col *= 256;
      f << int(col.e[0]) << " " << int(col.e[1]) << " " << int(col.e[2]) << "\n";
    }
  }
  f.close();
  std::cout << "Saved image." << std::endl;
}
