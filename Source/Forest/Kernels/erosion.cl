inline float rand(int2 co, int seed)
{
	float garbage = 0.f;
	return fract(sin(dot((float2)(co.x + seed, co.y + seed), (float2)(12.9898, 78.233))) * 43758.5453, &garbage);
}

// Take an input Water Height map and add a random amount of rain to it
inline void do_rainfall(
	__read_only image2d_t 	inWaterHeight,
	__write_only image2d_t 	outWaterHeight,
	uint					seed,
	float 					deltaTime,
	float					waterMul
	)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	// Calculate a random value
	float rainAmt 	= perlin2d((float)x, (float)y, 1.f / 1.f, 2, 4);
	// Multiply the value times the global mul and deltaTime
	rainAmt 		= rainAmt * waterMul * deltaTime;

	// Read the last frames value
	uint4 value = read_imageui(inWaterHeight, sampler, (int2)(x, y));
	write_imageui(outWaterHeight, (int2)(x, y), (uint) rainAmt + value.x);
}


void do_flux(
	__read_only image2d_t	inHeight,
	__read_only image2d_t 	inWaterHeight,
	__read_only image2d_t 	inFluxHeight,
	__write_only image2d_t 	outFluxHeight,
	float deltaTime)
{
	// fluxImg.x = fL
	// fluxImg.y = fR
	// fluxImg.z = fT
	// fluxImg.w = fB

	int x = get_global_id(0);
	int y = get_global_id(1);
	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	const float grav = 980.665f;
	const float area = 1.f;
	const float len = 1.f;

	float4 lastflux = read_imagef(inFluxHeight, sampler, (int2)(x, y));

	int4 height = convert_int4(read_imageui(inHeight, sampler, (int2)(x,y)).xxxx);
	int4 waterHeight = convert_int4(read_imageui(inWaterHeight, sampler, (int2)(x,y)).xxxx);

	int4 heightAdj =
	{
		read_imageui(inHeight, sampler, (int2)(x - 1,y)).x,
		read_imageui(inHeight, sampler, (int2)(x + 1,y)).x,
		read_imageui(inHeight, sampler, (int2)(x,y + 1)).x,
		read_imageui(inHeight, sampler, (int2)(x,y - 1)).x
	};

	int4 waterHeightAdj =
	{
		read_imageui(inWaterHeight, sampler, (int2)(x - 1,y)).x,
		read_imageui(inWaterHeight, sampler, (int2)(x + 1,y)).x,
		read_imageui(inWaterHeight, sampler, (int2)(x,y + 1)).x,
		read_imageui(inWaterHeight, sampler, (int2)(x,y - 1)).x
	};

	int4 heightDif = (waterHeight + height) - (waterHeightAdj + heightAdj);

	float4 fluxHeight = max((float4)(0.f, 0.f, 0.f, 0.f),
		lastflux + (deltaTime * area * ((grav * convert_float4(heightDif) / len))));

	write_imagef(outFluxHeight, (int2)(x, y), fluxHeight);
}

__kernel void erosion(
	__read_only image2d_t 	heightIn,		// 0
	__write_only image2d_t 	heightOut,		// 1
	__read_only image2d_t 	inWaterHeight,	// 2
	__write_only image2d_t 	outWaterHeight,	// 3
	__read_only image2d_t 	inSediment,		// 4
	__write_only image2d_t 	outSediment,	// 5
	__read_only image2d_t 	inFlux,			// 6
	__write_only image2d_t 	outFlux,		// 7
	__read_only image2d_t 	inVelocity,		// 8
	__write_only image2d_t 	outVelocity,	// 9
	uint					seed,			// 10
	uint 					iterations,		// 11
	float 					deltaTime,		// 12
	float					waterMul		// 13
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int h = get_image_height(heightIn);
	int w = get_image_width(heightIn);

	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	const float period = 32.f;

	uint4 value = read_imageui(heightIn, sampler, (int2)(x, y));

	do_rainfall(inWaterHeight, outWaterHeight, seed, deltaTime, waterMul);
	do_flux(heightIn, inWaterHeight, inFlux, outFlux, deltaTime);

	//value.x = (uint) ((sin((float)x / period) * cos((float)y / period) + 1.f) * 4096.f);

	//write_imageui(heightOut, (int2)(x, y), value);
}
