__kernel void multiply_by_scalar(
			__global float const* const src,
			__private float const coeff,
			__global float* const dst)
{
	uint const idx = get_global_id(0);

	dst[idx] = src[idx] * coeff;
}

