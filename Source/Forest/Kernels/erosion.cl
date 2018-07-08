inline float rand(int2 co, int seed)
{
	float garbage = 0.f;
	return fract(sin(dot((float2)(co.x + seed, co.y + seed), (float2)(12.9898, 78.233))) * 43758.5453, &garbage);
}

const sampler_t sampler = CLK_ADDRESS_NONE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

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

	// Calculate a random value
	float rainAmt 	= perlin2d((float)x, (float)y, 1.f / 1.f, 2, 4);
	// Multiply the value times the global mul and deltaTime
	rainAmt 		= rainAmt * waterMul * deltaTime;

	// Read the last frames value
	//float4 value = read_imagef(inWaterHeight, sampler, (int2)(x, y));
	//write_imagef(outWaterHeight, (int2)(x, y), rainAmt + value.x);

	write_imagef(outWaterHeight, (int2)(x, y), 100.f);
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

	const float grav = 0.980665f;
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
		((fluxAdj.x - flux.x) + (flux.y - fluxAdj.y)) / 2.f, // velocity x
		((fluxAdj.z - flux.z) + (flux.w - fluxAdj.w)) / 2.f, // velocity y
		0.f,
		0.f
	};

	write_imagef(outVelocity, (int2)(x, y), velocity);
}

inline float2 calculateNrm(
	__read_only image2d_t inHeight,
	bool bNormalize)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 heightAdj =
	{
		(float)read_imageui(inHeight, sampler, (int2)(x - 1,y)).x,
		(float)read_imageui(inHeight, sampler, (int2)(x + 1,y)).x,
		(float)read_imageui(inHeight, sampler, (int2)(x,y + 1)).x,
		(float)read_imageui(inHeight, sampler, (int2)(x,y - 1)).x
	};

	float height = (float)read_imageui(inHeight, sampler, (int2)(x, y)).x;

	// X+ Y+
	float2 nrm =
	{
		//(heightAdj.y - heightAdj.x) / 2.f,
		//(heightAdj.z - heightAdj.w) / 2.f,
		(heightAdj.x - height) + (height - heightAdj.y) / 2.f,
		(heightAdj.z - height) + (height - heightAdj.w) / 2.f
		//1.f
	};

	return bNormalize ? normalize(nrm) : nrm;
}

inline float calculateSinTiltAngle(
	__read_only image2d_t inHeight)
{
	float2 nrmVec = calculateNrm(inHeight, true);

	// This is the cos^2 of the tilt angle 
	//float vecCos = 1.f / (1.f + (nrmVec.x * nrmVec.x) + (nrmVec.y * nrmVec.y));
	float sin = sqrt((nrmVec.x * nrmVec.x) + (nrmVec.y * nrmVec.y)) / sqrt(1.f + (nrmVec.x * nrmVec.x) + (nrmVec.y * nrmVec.y));

	// sin = sqrt(1 - cos^2)
	//return vecCos;
	return sin;
}

inline float lmax(const float waterHeight, const float maxErosionDepth)
{
	if (waterHeight <= 0.f)
		return 0.f;
	else if (waterHeight >= maxErosionDepth)
		return 1.f;
	else // if (x > 0 && x < maxErosionDepth)
		return 1.f - ((maxErosionDepth - waterHeight) / maxErosionDepth);
}

__kernel void calculate_sediment_capacity(
	const float sedimentCapacity,
	const float maxErosionDepth,
	__read_only image2d_t	inHeight,
	__read_only image2d_t	inWaterHeight,
	__read_only image2d_t 	inVelocity,
	__write_only image2d_t	outSedimentCapacity)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float2 velocity = read_imagef(inVelocity, sampler, (int2)(x, y)).xy;
	uint waterHeight = read_imageui(inWaterHeight, sampler, (int2)(x, y)).x;

	float2 nrm = calculateNrm(inHeight, true);

	write_imagef(outSedimentCapacity, (int2)(x, y),
		//sedimentCapacity * calculateSinTiltAngle(inHeight) * velocityMagnitude * lmax((float)waterHeight, maxErosionDepth));
		sedimentCapacity * dot(-nrm, normalize(velocity)) * length(velocity) * lmax((float)waterHeight, maxErosionDepth));
		//length(velocity));
		//100.f);
		//(float4)(read_imagef(inVelocity, sampler, (int2)(x, y)).x, read_imagef(inVelocity, sampler, (int2)(x, y)).y, 0.f, 0.f));
		//(float4)(velocity.x, velocity.y, 0.f, 0.f));
}

float calculate_hardness_coefficient(
	__read_only image2d_t	inHardness,
	__read_only image2d_t	inSediment,
	__read_only image2d_t	inSedimentCapacity,
	const float sedimentCoefficient,
	const float softeningCoefficient,
	const float hardnessMin,
	const float deltaTime
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float hardness = read_imagef(inHardness, sampler, (int2)(x, y)).x;
	float sediment = read_imagef(inSediment, sampler, (int2)(x, y)).x;
	float sedimentCapacity = read_imagef(inSedimentCapacity, sampler, (int2)(x, y)).x;
	sedimentCapacity = 512.f;

	return max(hardnessMin, hardness - (deltaTime * softeningCoefficient * sedimentCoefficient * (sediment - sedimentCapacity)));
}

__kernel void calculate_erosion_deposition(
	__read_only image2d_t	inHeight,
	__write_only image2d_t	outHeight,
	__read_only image2d_t	inHardness,
	__read_only image2d_t	inSediment,
	__read_only image2d_t	inSedimentCapacity,
	float depositionSpeed,
	float sedimentCoefficient,
	float softeningCoefficient,
	float hardnessMin,
	float deltaTime
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float height = (float)read_imageui(inHeight, sampler, (int2)(x, y)).x;
	float hardness = read_imagef(inHardness, sampler, (int2)(x, y)).x;
	float sediment = read_imagef(inSediment, sampler, (int2)(x, y)).x;
	float sedimentCapacity = read_imagef(inSedimentCapacity, sampler, (int2)(x, y)).x;
	float hardnessCoefficient = calculate_hardness_coefficient(inHardness, inSediment, inSedimentCapacity, 
		sedimentCoefficient, softeningCoefficient, hardnessMin, deltaTime);

	if (sediment < sedimentCapacity)
	{	// bt+∆t = bt −∆t · Rt(x, y) · Ks(C − st)
		height = height - (deltaTime * hardnessCoefficient * sedimentCoefficient * (sedimentCapacity - sediment));
		//height = hardnessCoefficient;//(deltaTime * hardnessCoefficient * sedimentCoefficient * (sedimentCapacity - sediment));
		//height = 0.f;
	}
	else
	{	// bt+∆t = bt + ∆t · Kd(st −C)
		height = height + (deltaTime * depositionSpeed * (sediment - sedimentCapacity));
		//height = sediment - sedimentCapacity; // (deltaTime * depositionSpeed * (sediment - sedimentCapacity));
		//height = 0.f;
	}

	height = clamp(height, 0.f, 65535.f);

	//height = sedimentCapacity;

	//height = hardnessCoefficient;

	write_imageui(outHeight, (int2)(x, y), convert_uint(height));
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