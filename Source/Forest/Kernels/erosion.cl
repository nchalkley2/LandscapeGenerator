inline float rand(int2 co, int seed)
{
	float garbage = 0.f;
	return fract(sin(dot((float2)(co.x + seed, co.y + seed), (float2)(12.9898, 78.233))) * 43758.5453, &garbage);
}

// Take an input Water Height map and add a random amount of rain to it
__kernel void rainfall(
	__read_only image2d_t 	inWaterHeight, // These can be the same value
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
	float4 value = read_imagef(inWaterHeight, sampler, (int2)(x, y));
	write_imagef(outWaterHeight, (int2)(x, y), rainAmt + value.x);
}


__kernel void flux(
	__read_only image2d_t	inHeight,
	__read_only image2d_t 	inWaterHeight,
	__read_only image2d_t 	inFluxHeight, // These have to be different
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

	uint4 height = convert_uint4(read_imageui(inHeight, sampler, (int2)(x,y)).xxxx);
	float4 waterHeight = convert_float4(read_imagef(inWaterHeight, sampler, (int2)(x,y)).xxxx);

	uint4 heightAdj =
	{
		read_imageui(inHeight, sampler, (int2)(x - 1,y)).x,
		read_imageui(inHeight, sampler, (int2)(x + 1,y)).x,
		read_imageui(inHeight, sampler, (int2)(x,y + 1)).x,
		read_imageui(inHeight, sampler, (int2)(x,y - 1)).x
	};

	float4 waterHeightAdj =
	{
		read_imagef(inWaterHeight, sampler, (int2)(x - 1,y)).x,
		read_imagef(inWaterHeight, sampler, (int2)(x + 1,y)).x,
		read_imagef(inWaterHeight, sampler, (int2)(x,y + 1)).x,
		read_imagef(inWaterHeight, sampler, (int2)(x,y - 1)).x
	};

	float4 heightDif = (waterHeight + convert_float4(height)) - (waterHeightAdj + convert_float4(heightAdj));

	float4 fluxHeight = max((float4)(0.f, 0.f, 0.f, 0.f),
		lastflux + (deltaTime * area * ((grav * convert_float4(heightDif) / len))));
		//(deltaTime * area * ((grav * convert_float4(heightDif) / len))));

	write_imagef(outFluxHeight, (int2)(x, y), fluxHeight);
}

__kernel void calculate_k_factor(
	__read_only image2d_t 	inWaterHeight,
	__read_only image2d_t 	inFluxHeight,
	__write_only image2d_t 	outFluxHeight,
	float deltaTime)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	const float len = 1.f;

	uint waterHeight = read_imageui(inWaterHeight, sampler, (int2)(x, y)).x;
	float4 flux = read_imagef(inFluxHeight, sampler, (int2)(x, y));

	// This is the scaling factor for the flux. If the flux's magnitude is too large it will scale it down
	float K = min(1.f, (convert_float(waterHeight) * len) / ((flux.x + flux.y + flux.z + flux.w) * deltaTime));

	flux *= K;

	write_imagef(outFluxHeight, (int2)(x, y), flux);
	//write_imagef(outFluxHeight, (int2)(x, y), (float4)(1.0f, 1.0f, 0.0f, 1.0f));
}

__kernel void calculate_water_height_change(
	__read_only image2d_t 	inWaterHeight,
	__write_only image2d_t 	outWaterHeight,
	__read_only image2d_t 	inFluxHeight,
	float deltaTime)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	const float len = 1.f;

	// fluxImg.x = fL
	// fluxImg.y = fR
	// fluxImg.z = fT
	// fluxImg.w = fB

	// Read the adjacent flux cells and add them
	float fluxIn =
		  read_imagef(inFluxHeight, sampler, (int2)(x - 1, y)).y	// Right
		+ read_imagef(inFluxHeight, sampler, (int2)(x + 1, y)).x	// Left
		+ read_imagef(inFluxHeight, sampler, (int2)(x, y + 1)).w	// Top
		+ read_imagef(inFluxHeight, sampler, (int2)(x, y - 1)).z;	// Bottom

	// Add all the values in the current flux cell
	float4 flux = read_imagef(inFluxHeight, sampler, (int2)(x, y));
	float fluxOut = flux.x + flux.y + flux.z + flux.w;

	float waterDif = (fluxIn - fluxOut) * deltaTime;
	
	// Read the water input, add the water difference, and write out
	int waterHeight = read_imageui(inWaterHeight, sampler, (int2)(x, y)).x;
	write_imageui(outWaterHeight, (int2)(x, y), (uint)(waterHeight + convert_int(waterDif / len)));
}

__kernel void calculate_velocity(
	__read_only image2d_t 	inFluxHeight,
	__write_only image2d_t	outVelocity,
	float deltaTime)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	const float len = 1.f;

	// fluxImg.x = fL
	// fluxImg.y = fR
	// fluxImg.z = fT
	// fluxImg.w = fB

	// velocityImg.x = fL
	// velocityImg.y = fR
	// velocityImg.z = fT
	// velocityImg.w = fB

	float4 flux = read_imagef(inFluxHeight, sampler, (int2)(x, y));

	float4 fluxAdj =
	{
		read_imagef(inFluxHeight, sampler, (int2)(x - 1,y)).y,
		read_imagef(inFluxHeight, sampler, (int2)(x + 1,y)).x,
		read_imagef(inFluxHeight, sampler, (int2)(x,y + 1)).w,
		read_imagef(inFluxHeight, sampler, (int2)(x,y - 1)).z
	};

	float4 velocity =
	{
		(fluxAdj.x - flux.x + flux.y - fluxAdj.y) / 2.f, // velocity x
		(fluxAdj.z - flux.z + flux.w - fluxAdj.w) / 2.f, // velocity y
		0.0,
		0.0
	};

	write_imagef(outVelocity, (int2)(x, y), velocity);
}

inline float2 calculateNrm(
	__read_only image2d_t inHeight)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	float4 heightAdj =
	{
		(float)read_imageui(inHeight, sampler, (int2)(x - 1,y)).x,
		(float)read_imageui(inHeight, sampler, (int2)(x + 1,y)).x,
		(float)read_imageui(inHeight, sampler, (int2)(x,y + 1)).x,
		(float)read_imageui(inHeight, sampler, (int2)(x,y - 1)).x
	};

	float height = (float)read_imageui(inHeight, sampler, (int2)(x, y)).x;

	// X+ Y+
	float2 nrmVec =
	{
		(heightAdj.x - height) + (height - heightAdj.y),
		(heightAdj.z - height) + (height - heightAdj.w)
	};

	return normalize(nrmVec);
}

inline float calculateSinTiltAngle(
	__read_only image2d_t inHeight)
{
	float2 nrmVec = calculateNrm(inHeight);

	// This is the cos^2 of the tilt angle 
	float vecCos = 1.0 / (1.0 + nrmVec.x * nrmVec.x + nrmVec.y * nrmVec.y);

	// sin = sqrt(1 - cos^2)
	return sqrt(1.0 - vecCos);
}

inline float lmax(const float waterHeight, const float maxErosionDepth)
{
	if (waterHeight <= 0)
		return 0.0;
	else if (waterHeight >= maxErosionDepth)
		return 1.0;
	else // if (x > 0 && x < maxErosionDepth)
		return 1.0 - ((maxErosionDepth - waterHeight) / maxErosionDepth);
}

float calculate_sediment_capacity(
	const float sedimentCapacity,
	__read_only image2d_t	inHeight,
	__read_only image2d_t 	inVelocity)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

	float velocityMagnitude = length(read_imagef(inVelocity, sampler, (int2)(x, y)).xy);

	return sedimentCapacity * calculateSinTiltAngle(inHeight) * velocityMagnitude;
}

/*
__kernel void erosion(
	__read_only image2d_t 	heightIn,		// 0
	__write_only image2d_t 	heightOut,		// 1
	__read_only image2d_t 	inWaterHeight,	// 2
	__write_only image2d_t 	outWaterHeight,	// 3
	//__read_only image2d_t 	inSediment,		// 4
	//__write_only image2d_t 	outSediment,	// 5
	__read_only image2d_t 	inFlux,			// 6
	__write_only image2d_t 	outFlux,		// 7
	//__read_only image2d_t 	inVelocity,		// 8
	//__write_only image2d_t 	outVelocity,	// 9
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
}
*/