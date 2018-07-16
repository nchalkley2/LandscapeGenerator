inline float rand(int2 co, int seed)
{
	float garbage = 0.f;
	return fract(sin(dot((float2)(co.x + seed, co.y + seed), (float2)(12.9898, 78.233))) * 43758.5453, &garbage);
}

const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_NEAREST | CLK_ADDRESS_CLAMP_TO_EDGE;

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
	float rainAmt	= read_imagef(inWaterHeight, sampler, (int2)(x, y)).x;//perlin2d((float)x, (float)y, 1.f / 1.f, 2, 4);
	// Multiply the value times the global mul and deltaTime
	rainAmt 		+= waterMul * deltaTime;

	// Read the last frames value
	//float4 value = read_imagef(inWaterHeight, sampler, (int2)(x, y));
	//write_imagef(outWaterHeight, (int2)(x, y), rainAmt + value.x);
	write_imagef(outWaterHeight, (int2)(x, y), rainAmt);
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
	// fluxImg.z = fB
	// fluxImg.w = fT

	int x = get_global_id(0);
	int y = get_global_id(1);

	const float grav = 9.80665f;
	const float area = 20.f;
	const float len = 5.f;

	float4 lastflux = read_imagef(inFluxHeight, sampler, (int2)(x, y));

	float4 height = read_imagef(inHeight, sampler, (int2)(x,y)).xxxx;
	float4 waterHeight = read_imagef(inWaterHeight, sampler, (int2)(x,y)).xxxx;

	float4 heightAdj =
	{
		read_imagef(inHeight, sampler, (int2)(x - 1,y)).x,
		read_imagef(inHeight, sampler, (int2)(x + 1,y)).x,
		read_imagef(inHeight, sampler, (int2)(x,y - 1)).x,
		read_imagef(inHeight, sampler, (int2)(x,y + 1)).x
	};

	float4 waterHeightAdj =
	{
		read_imagef(inWaterHeight, sampler, (int2)(x - 1,y)).x,
		read_imagef(inWaterHeight, sampler, (int2)(x + 1,y)).x,
		read_imagef(inWaterHeight, sampler, (int2)(x,y - 1)).x,
		read_imagef(inWaterHeight, sampler, (int2)(x,y + 1)).x
	};

	float4 heightDif = (waterHeight + height) - (waterHeightAdj + heightAdj);

	float4 fluxHeight = max((float4)(0.f, 0.f, 0.f, 0.f),
		lastflux + (deltaTime * area * ((grav * heightDif) / len)));

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

	const float len = 5.f;

	float waterHeight = read_imagef(inWaterHeight, sampler, (int2)(x, y)).x;
	float4 flux = read_imagef(inFluxHeight, sampler, (int2)(x, y));

	// This is the scaling factor for the flux. If the flux's magnitude is too large it will scale it down
	// This is SUPPOSED to be min. The paper cited in this source code is wrong
	float fluxAdd = flux.x + flux.y + flux.z + flux.w;
	fluxAdd = max(fluxAdd, 0.001f);

	float K = min(1.f, (waterHeight * len) / ((fluxAdd) * deltaTime));

	flux *= K;

	write_imagef(outFluxHeight, (int2)(x, y), flux);
}

__kernel void calculate_water_height_change(
	__read_only image2d_t 	inWaterHeight,
	__write_only image2d_t 	outWaterHeight,
	__read_only image2d_t 	inFluxHeight,
	float deltaTime)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	const float len = 5.f;

	// fluxImg.x = fL
	// fluxImg.y = fR
	// fluxImg.z = fB
	// fluxImg.w = fT

	// Read the adjacent flux cells and add them
	float fluxIn =
		  read_imagef(inFluxHeight, sampler, (int2)(x - 1, y)).y	// Left
		+ read_imagef(inFluxHeight, sampler, (int2)(x + 1, y)).x	// Right
		+ read_imagef(inFluxHeight, sampler, (int2)(x, y - 1)).w	// Bottom
		+ read_imagef(inFluxHeight, sampler, (int2)(x, y + 1)).z;	// Top

	// Add all the values in the current flux cell
	float4 flux = read_imagef(inFluxHeight, sampler, (int2)(x, y));
	float fluxOut = flux.x + flux.y + flux.z + flux.w;

	float waterDif = (fluxIn - fluxOut) * deltaTime;
	
	// Read the water input, add the water difference, and write out
	float waterHeight = read_imagef(inWaterHeight, sampler, (int2)(x, y)).x;
	write_imagef(outWaterHeight, (int2)(x, y), waterHeight + (waterDif / (len * len)));
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
	// fluxImg.z = fB
	// fluxImg.w = fT

	// velocityImg.x = fL
	// velocityImg.y = fR
	// velocityImg.z = fB
	// velocityImg.w = fT

	float4 flux = read_imagef(inFluxHeight, sampler, (int2)(x, y));

	float4 fluxAdj =
	{
		read_imagef(inFluxHeight, sampler, (int2)(x - 1,y)).y,
		read_imagef(inFluxHeight, sampler, (int2)(x + 1,y)).x,
		read_imagef(inFluxHeight, sampler, (int2)(x,y - 1)).w,
		read_imagef(inFluxHeight, sampler, (int2)(x,y + 1)).z
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
		read_imagef(inHeight, sampler, (int2)(x - 1,y)).x,
		read_imagef(inHeight, sampler, (int2)(x + 1,y)).x,
		read_imagef(inHeight, sampler, (int2)(x,y - 1)).x,
		read_imagef(inHeight, sampler, (int2)(x,y + 1)).x
	};

	float height = read_imagef(inHeight, sampler, (int2)(x, y)).x;

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

/*inline float calculateSinTiltAngle(
	__read_only image2d_t inHeight)
{
	float2 nrmVec = calculateNrm(inHeight, true);

	// This is the cos^2 of the tilt angle 
	//float vecCos = 1.f / (1.f + (nrmVec.x * nrmVec.x) + (nrmVec.y * nrmVec.y));
	float sin = sqrt((nrmVec.x * nrmVec.x) + (nrmVec.y * nrmVec.y)) / sqrt(1.f + (nrmVec.x * nrmVec.x) + (nrmVec.y * nrmVec.y));

	// sin = sqrt(1 - cos^2)
	//return vecCos;
	return sin;
}*/

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
	float waterHeight = read_imagef(inWaterHeight, sampler, (int2)(x, y)).x;

	float2 nrm = calculateNrm(inHeight, true);

	write_imagef(outSedimentCapacity, (int2)(x, y),
		//C(x, y) = Kc ·(-N(x, y)·V)· |v(x, y)| · lmax(d1(x, y))
		//sedimentCapacity * dot(-nrm, normalize(velocity)) * length(velocity) * lmax(waterHeight, maxErosionDepth));
		sedimentCapacity * length(velocity) * lmax(waterHeight, maxErosionDepth));
		//lmax(waterHeight, maxErosionDepth) * length(velocity) * 1.f);

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

	// Rt+∆t(x, y) = max(Rmin,Rt(x, y) − (∆t ·Kh * Ks(st −C)))
	return max(hardnessMin, hardness - (deltaTime * softeningCoefficient * sedimentCoefficient * (sediment - sedimentCapacity)));
}

__kernel void calculate_erosion_deposition(
	__read_only image2d_t	inHeight,
	__write_only image2d_t	outHeight,
	__read_only image2d_t	inHardness,
	__read_only image2d_t	inSediment,
	__write_only image2d_t	outSediment,
	__read_only image2d_t	inSedimentCapacity,
	__read_only image2d_t	inWaterHeight,
	__write_only image2d_t	outWaterHeight,

	float depositionSpeed,
	float sedimentCoefficient,
	float softeningCoefficient,
	float hardnessMin,
	float deltaTime
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float height =				read_imagef(inHeight, sampler, (int2)(x, y)).x;
	float waterHeight =			read_imagef(inWaterHeight, sampler, (int2)(x, y)).x;
	float hardness =			read_imagef(inHardness, sampler, (int2)(x, y)).x;
	float sediment =			read_imagef(inSediment, sampler, (int2)(x, y)).x;
	float sedimentCapacity =	read_imagef(inSedimentCapacity, sampler, (int2)(x, y)).x;
	float hardnessCoefficient =	calculate_hardness_coefficient(inHardness, inSediment, inSedimentCapacity, 
		sedimentCoefficient, softeningCoefficient, hardnessMin, deltaTime);

	float diff = 0.f;

	if (sediment < sedimentCapacity)
	{
		// ∆t · Rt(x, y) · Ks(C − st)
		diff = (deltaTime * hardnessCoefficient * sedimentCoefficient * (sedimentCapacity - sediment));

		// bt+∆t = bt − ∆t · Rt(x, y) · Ks(C − st)
		height -= diff;
		// s1 = st + ∆t · Rt(x, y) · Ks(C − st)
		sediment += diff;
		// d3 = d2 + ∆t · Rt(x, y) · Ks(C − st)
		waterHeight += diff;
	}
	else
	{	
		// ∆t · Kd(st −C)
		diff = (deltaTime * depositionSpeed * (sediment - sedimentCapacity));

		// bt+∆t = bt + ∆t · Kd(st − C)
		height += diff;
		// s1 = st − ∆t · Kd(st − C)
		sediment -= diff;
		// d3 = d2 − ∆t · Kd(st − C)
		waterHeight -= diff;
	}

	write_imagef(outWaterHeight, (int2)(x, y), waterHeight);
	write_imagef(outSediment, (int2)(x, y), sediment);
	write_imagef(outHeight, (int2)(x, y), height);
}

__kernel void move_sediment(
	__read_only image2d_t inSediment,
	__write_only image2d_t outSediment,
	__read_only image2d_t inVelocity,

	float deltaTime
)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	float2 uv = read_imagef(inVelocity, sampler, (int2)(x, y)).xy * deltaTime;

	float2 coord = (float2)(x, y) - uv;

	// linearly interpolate between samples since uv can sample inbetween pixels
	const sampler_t sedimentSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_FILTER_LINEAR | CLK_ADDRESS_CLAMP_TO_EDGE;
	float sediment = read_imagef(inSediment, sedimentSampler, coord).x;
	write_imagef(outSediment, (int2)(x,y), sediment);
}
