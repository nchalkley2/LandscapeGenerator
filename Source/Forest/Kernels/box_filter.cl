__kernel void box_filter(__read_only image2d_t input,
		__write_only image2d_t output,
		uint box_height,
		uint box_width)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int h = get_image_height(input);
	int w = get_image_width(input);
	int k = box_width;
	int l = box_height;

	/*if (x < k / 2 || y < l / 2 || x >= w - (k / 2) || y >= h - (l / 2)) {
		write_imagef(output, (int2)(x, y), (float4)(0, 0, 0, 1));
	}
	else {
		const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

		float4 sum = { 0, 0, 0, 1 };
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < l; j++) {
				sum += read_imagef(input, sampler, (int2)(x + i - k, y + j - l));
			}
		}
		sum /= (float)k * l;
		float4 value = (float4)(sum.x, 0.f, 0.f, 1.f);
		write_imagef(output, (int2)(x, y), value);
	}*/

	if (x < k / 2 || y < l / 2 || x >= w - (k / 2) || y >= h - (l / 2)) {
		write_imageui(output, (int2)(x, y), (uint4)(0, 0, 0, 1));
	}
	else {
		const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

		uint4 sum = { 0, 0, 0, 1 };
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < l; j++) {
				sum += read_imageui(input, sampler, (int2)(x + i - k, y + j - l));
			}
		}
		sum /= (uint)k * l;
		uint4 value = (uint4)(sum.x, 0.f, 0.f, 1.f);
		write_imageui(output, (int2)(x, y), value);
	}
	
	/*const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	uint4 value = (uint4)(0, 0, 0, 0);
	value = read_imageui(input, sampler, (int2)(x, y));
	value = value + (uint4)(x, 0, 0, 0);
	write_imageui(output, (int2)(x, y), value);*/

	/*const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;
	float4 value = (float4)(0, 0, 0, 0);
	value = read_imagef(input, sampler, (int2)(x, y));
	value = value + (float4)(x, 0, 0, 0);
	write_imagef(output, (int2)(x, y), value);*/
}