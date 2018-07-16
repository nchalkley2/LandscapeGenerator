inline float rand(int2 co, int seed)
{
	float garbage = 0.f;
	return fract(sin(dot((float2)(co.x + seed, co.y + seed), (float2)(12.9898, 78.233))) * 43758.5453, &garbage);
}

// xy_dim: width, height of the image
inline float2 getPoint(int2 xy, int2 xy_dim, int size, int seed)
{
	int2 dxy = xy / size;

	return (float2)(dxy.x * size, dxy.y * size) + ((float2)(rand(dxy, seed), rand(dxy + xy_dim, seed)) * (float2)(size, size));
}

inline float getDist(int x1, int y1, int x2, int y2, int w, int h, int size, int seed)
{
	return distance((float2)(x1, y1), getPoint((int2)(x2, y2), (int2)(w, h), size, seed));
}

// Helper func sorta
inline float multi_min(float x1, float x2, float x3, float x4, float x5, float x6, float x7, float x8, float x9)
{
	return min(x1, min(x2, min(x3, min(x4, min(x5, min(x6, min(x7, min(x8, x9))))))));
}

inline void xorSwap(float* x, float* y)
{
	// Bitwise shit doesn't work on floats for some reason...
	*((int*)x) ^= *((int*)y);
	*((int*)y) ^= *((int*)x);
	*((int*)x) ^= *((int*)y);
}

inline void sort_asc(float* arr, int num)
{
	int j, i;
	for (i = 1; i < num; i++) {
		for (j = 0; j < num - i; j++) {
			if (arr[j] > arr[j + 1]) {
				xorSwap(&arr[j], &arr[j + 1]);
			}
		}
	}
}

// second most minmum
inline float get_f2(float x1, float x2, float x3, float x4, float x5, float x6, float x7, float x8, float x9)
{
	float xarr[] = { x1, x2, x3, x4, x5, x6, x7, x8, x9 };
	sort_asc(xarr, 9);

	return xarr[0];//xarr[1] - xarr[0];
}

__kernel void voronoi(__read_only image2d_t heightIn,
	__write_only image2d_t heightOut,
	int size,
	int seed,
	float amplitude)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int h = get_image_height(heightIn);
	int w = get_image_width(heightIn);

	float randPoint = getDist(x, y, x, y, w, h, size, seed);

	// adjacent points
	// x+, x-, y+, y-
	// x+ y+, x- y+, x+ y-, x- y-
	float8 adjRandPoints = {
		getDist(x, y, x + size, y, w, h, size, seed),
		getDist(x, y, x - size, y, w, h, size, seed),
		getDist(x, y, x, y + size, w, h, size, seed),
		getDist(x, y, x, y - size, w, h, size, seed),
		getDist(x, y, x + size, y + size, w, h, size, seed),
		getDist(x, y, x - size, y + size, w, h, size, seed),
		getDist(x, y, x + size, y - size, w, h, size, seed),
		getDist(x, y, x - size, y - size, w, h, size, seed)
	};

	float out = multi_min(
		randPoint,
		adjRandPoints.s0,
		adjRandPoints.s1,
		adjRandPoints.s2,
		adjRandPoints.s3,
		adjRandPoints.s4,
		adjRandPoints.s5,
		adjRandPoints.s6,
		adjRandPoints.s7) * (amplitude / size);

	//out = (uint) amplitude;

	write_imagef(heightOut, (int2)(x, y), out);
}