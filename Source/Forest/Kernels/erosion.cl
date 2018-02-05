inline float rand(int2 co, int seed)
{
	float garbage = 0.f;
	return fract(sin(dot((float2)(co.x + seed, co.y + seed), (float2)(12.9898, 78.233))) * 43758.5453, &garbage);
}

inline void do_rainfall(
	__read_only image2d_t 	inWaterHeight,
	__write_only image2d_t 	outWaterHeight,
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

/*
void do_flux(float deltaTime, __read_only image2d_t heightImg, __write_only image2d_t waterHeightImg, image2d_t fluxImg)
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

	float4 lastflux = read_imagef(fluxImg, sampler, (int2)(x, y));

	int4 height = convert_int4(read_imageui(heightImg, sampler, (int2)(x,y)).xxxx);
	int4 waterHeight = convert_int4(read_imageui(waterHeightImg, sampler, (int2)(x,y)).xxxx);

	int4 heightAdj =
	{
		read_imageui(heightImg, sampler, (int2)(x - 1,y)).x,
		read_imageui(heightImg, sampler, (int2)(x + 1,y)).x,
		read_imageui(heightImg, sampler, (int2)(x,y + 1)).x,
		read_imageui(heightImg, sampler, (int2)(x,y - 1)).x
	};

	int4 waterHeightAdj =
	{
		read_imageui(waterHeightImg, sampler, (int2)(x - 1,y)).x,
		read_imageui(waterHeightImg, sampler, (int2)(x + 1,y)).x,
		read_imageui(waterHeightImg, sampler, (int2)(x,y + 1)).x,
		read_imageui(waterHeightImg, sampler, (int2)(x,y - 1)).x
	};

	int4 heightDif = (waterHeight + height) - (waterHeightAdj + heightAdj);

	fluxImg.x = max(0, lastflux.x + (deltaTime * area * (g * )));
}
*/

__kernel void erosion(
	__read_only image2d_t 	heightIn,
	__write_only image2d_t 	heightOut,
	__read_only image2d_t 	inWaterHeight,
	__write_only image2d_t 	outWaterHeight,
	__read_only image2d_t 	inSediment,
	__write_only image2d_t 	outSediment,
	__read_only image2d_t 	inFlux,
	__write_only image2d_t 	outFlux,
	__read_only image2d_t 	inVelocity,
	__write_only image2d_t 	outVelocity,
	uint 					iterations,
	float 					deltaTime,
	float					waterMul
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int h = get_image_height(heightIn);
	int w = get_image_width(heightIn);

	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	const float period = 32.f;

	uint4 value = read_imageui(heightIn, sampler, (int2)(x, y));

	do_rainfall(inWaterHeight, outWaterHeight, waterMul);
	
	//parabolid(heightOut);

	//value.x = (uint) ((sin((float)x / period) * cos((float)y / period) + 1.f) * 4096.f);

	//write_imageui(heightOut, (int2)(x, y), value);
}
