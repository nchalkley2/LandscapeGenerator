// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Core.h"

// Disable warning for GNU_C not being defined
#pragma warning(push)
#pragma warning(disable: 4668)
#define BOOST_COMPUTE_THREAD_SAFE
#define BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
#define BOOST_DISABLE_ABI_HEADERS
#include <boost/compute/system.hpp>
#include <boost/compute/image/image2d.hpp>
#include <boost/compute/utility/dim.hpp>
#include <boost/compute/utility/source.hpp>
#pragma warning(pop)

#include <vector>
#include <mutex>
#include <string>
#include <memory>

#include "CoreMinimal.h"
#include "LandscapeGeneration.generated.h"

UENUM(BlueprintType)		//"BlueprintType" is essential to include
enum class EMixType : uint8
{
	E_Add			= 0	UMETA(DisplayName = "Add"),
	E_Subtract		= 1	UMETA(DisplayName = "Subtract"),
	E_Multiply		= 2	UMETA(DisplayName = "Multiply"),
	E_Min			= 3 UMETA(DisplayName = "Minimum"),
	E_Max			= 4 UMETA(DisplayName = "Maxmimum")
};

namespace LandscapeGeneration
{
	extern boost::compute::image_format ImageFormat;

	class Heightmap
	{
	public:
		Heightmap(
			int SizeX, 
			int SizeY, 
			boost::compute::image_format inImageFormat
				= ImageFormat
		);

		// Copy the heightmap from the device to the client in a TArray<uint16>
		operator TArray<uint16>() const;

		boost::compute::image2d Image;
	};

	// Make sure you call SetDevices to initialize the module
	void SetDevices(std::vector<boost::compute::device> Devices);

	// Creates a heightmap on the device and returns a wrapper pointer to it
	std::shared_ptr<Heightmap> CreateHeightmap(
		int SizeX, 
		int SizeY, 
		boost::compute::image_format inImageFormat 
			= ImageFormat
	);

	void PushKernel(std::function<void()> KernelFunc);

	// Manages the kernel thread. Should be ticked from UE4's game thread
	void Tick();

	namespace Kernels
	{
		void PerlinNoise(boost::compute::image2d& Heightmap,
			float noiseSize, int32 seed, int32 depth, float amplitude);

		void WarpedPerlinNoise(boost::compute::image2d& Heightmap,
			float noiseSize, int32 seed, int32 depth, float amplitude);

		void Mix(boost::compute::image2d& LHeightMap,
			boost::compute::image2d& RHeightMap,
			boost::compute::image2d& OutputHeightmap,
			EMixType MixType);

		void VoronoiNoise(boost::compute::image2d& Heightmap,
			int32 noiseSize,
			int32 seed,
			float amplitude);

		void Constant(boost::compute::image2d& Heightmap,
			int32 height);

		void Erosion(boost::compute::image2d& Heightmap);
	}
}
