__kernel void multiply_by_scalar(
              __private float coeff,
              __global float * src,
              __global float*  res)
  {
    uint const idx = get_global_id(0);
    res[idx] = src[idx] * coeff;
  }

  __kernel void fill_vbo(__global float* vbo){
    int id=get_global_id(0);
    vbo[id]=(id%6)/3+(id%2)*(id/6);
    vbo[id]/=3;
  }
