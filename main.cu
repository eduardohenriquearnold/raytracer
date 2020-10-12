#include <iostream>
#include <fstream>
#include "ray.h"
#include "camera.h"
#include "hitable.h"
#include "material.h"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__device__ vec3 color(const ray& r, hitable** h, curandState *s) {
  ray cur_ray = r;
  vec3 cur_attenuation = vec3(1.,1.,1.);
  
  for (int iter=0; iter<50; iter++)
  {
    hit_record rec;
    if ((**h).hit(cur_ray, 0.001f, 1e20f, rec)) {
      ray scattered;
      vec3 attenuation;
      if (rec.mat_ptr->scatter(cur_ray, rec, attenuation, scattered, s)) {
        cur_attenuation *= attenuation;
        cur_ray = scattered;
      }
      else 
        return vec3(0,0,0);
    }
    else{
      //background
      vec3 unit_direction = unit_vector(r.direction());
      float t = 0.5f*(unit_direction.y() + 1.0f);
      vec3 c = (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
      return cur_attenuation*c;
    }
  }

  //if exceeds max iteration (number of reflections)
  return vec3(0,0,0);
}

__global__ void render(vec3 *fb, const int nx, const int ny, const int ns, camera** cam, hitable** h, curandState *crs) {
  int i = blockIdx.x * blockDim.x +threadIdx.x;
  int j = blockIdx.y * blockDim.y +threadIdx.y;
  int pixelIdx = i + j*nx;
  if (i>=nx || j>=ny)
    return;
  curandState ls = crs[pixelIdx];

  vec3 col(0,0,0);
  for (int s=0; s<ns; s++)
  {
    float u = (i +random_float(&ls))/ float(nx);
    float v = (j +random_float(&ls)) / float(ny);
    ray r = (**cam).get_ray(u,v);
    col += color(r, h, &ls);
  }
  col /= ns;

  //gamma correction
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

__global__ void init_cam(camera** c, int nx, int ny){
  if (!(threadIdx.x == 0 && blockIdx.x == 0))
    return;

  // camera *cam = new camera(vec3(13,2,3), vec3(0,0,0), vec3(0,1,0), 20, float(nx)/float(ny));
  *c = new camera(vec3(0,0,0), vec3(0,0,-1), vec3(0,1,0), 90, float(nx)/ny);
}

__global__ void init_world(hitable** w){
  if (!(threadIdx.x == 0 && blockIdx.x == 0))
    return;

  *w = new hitable_list(10);

  hitable_list* world = (hitable_list*) *w;
  world->add(new sphere(vec3(0,-100.5,-1), 100.f, new lambertian(vec3(0.3,0.3,0.3))));
  world->add(new sphere(vec3(0,0,-1), 0.5f, new lambertian(vec3(0.3,0.5,0.3)) ));
}

int main()
{
  std::ofstream f;
  f.open("output.ppm", std::ios::out);

  //define rendering limits/properties
  const int nx = 600;
  const int ny = 400;
  const int ns = 1000;

  //define thread numbers and block dimensions
  const int tx = 8;
  const int ty = 8;
  const dim3 blocks(nx/tx+1, ny/ty+1);
  const dim3 threads(tx,ty);

  //allocate device memory for image
  vec3 *fb;
  cudaMallocManaged(&fb, sizeof(vec3)*nx*ny);

  //create random states
  curandState *crs;
  cudaMalloc(&crs, nx*ny*sizeof(curandState));
  init_random_states<<<blocks,threads>>>(crs, nx, ny);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());

  //create camera
  camera **cam;
  cudaMalloc(&cam, sizeof(camera*));
  init_cam<<<1,1>>>(cam, nx, ny);
  cudaDeviceSynchronize();
  
  //create world (objects and materials) in device
  hitable** world;
  cudaMalloc(&world, sizeof(hitable*));
  init_world<<<1,1>>>(world);
  cudaDeviceSynchronize();

  //render world
  std::cout << "Started rendering..." << std::endl;
  render<<<blocks, threads>>>(fb, nx, ny, ns, cam, world, crs);
  cudaDeviceSynchronize();
  checkCudaErrors(cudaGetLastError());
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
  cudaFree(crs);
  cudaFree(cam);
  cudaFree(world);
}
