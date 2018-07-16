#include "perlin.h"

__kernel void warpedperlin(__read_only image2d_t heightIn,
	__write_only image2d_t heightOut,
	float size,
	int seed,
	int depth,
	float amplitude)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float2 warpedcoords = (float2)(perlin2d((float)x, (float)y, 1.f / size, depth, seed),
								 perlin2d((float)x + 5.2f, (float)y + 1.3f, 1.f / size, depth, seed));
	warpedcoords *= 256.f;

	float out = perlin2d((float)x + warpedcoords.x, (float)y + warpedcoords.y, 1.f / (size + warpedcoords.x), depth, seed) * amplitude;

	write_imagef(heightOut, (int2)(x, y), out);
}
