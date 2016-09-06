__constant sampler_t sampler_const =
CLK_NORMALIZED_COORDS_FALSE |
CLK_ADDRESS_CLAMP |
CLK_FILTER_NEAREST;

__kernel void rgb2gray_unrolled(read_only image2d_t source, write_only image2d_t dest) {
	const int2 pixel_id = (int2)(get_global_id(0), get_global_id(1));
	const float4 rgba = read_imagef(source, sampler_const, pixel_id);
	const float gray = 0.2126 * rgba.x + 0.7152 * rgba.y + 0.0722 * rgba.z;
	write_imagef(dest, pixel_id, (float4)(gray, gray, gray, 1.0));
}

__kernel void rgb2gray_patches(const int batch_size, read_only image2d_t source, write_only image2d_t dest) {
	const int2 group = (int2)(batch_size, batch_size);
	const int2 base = (int2)(get_global_id(0) * group.x, get_global_id(1) * group.y);

	for (int i = 0; i < group.x; i++) {
		for (int j = 0; j < group.y; j++) {
			const int2 loco = base + (int2)(i, j);
			const float4 rgba = read_imagef(source, sampler_const, loco);
			const float gray = 0.2126 * rgba.x + 0.7152 * rgba.y + 0.0722 * rgba.z;
			write_imagef(dest, loco, (float4)(gray, gray, gray, 1.0));
		}
	}
}

