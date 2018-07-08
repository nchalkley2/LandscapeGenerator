// Fill out your copyright notice in the Description page of Project Settings.

#include "LandscapeGeneration.h"

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
#include <boost/compute/container/vector.hpp>
#pragma warning(pop)

#include <memory>
#include <vector>
#include <queue>
#include <atomic>
#include <array>

#include "LandscapeGeneration.inl"

#include <io.h>  
#include <stdlib.h>  
#include <cstdio>
#include <cstring>

using namespace std;
namespace compute = boost::compute;

namespace LandscapeGeneration
{
	static vector<compute::device>				Devices;
	static unique_ptr<compute::context>			Context;
	static unique_ptr<compute::command_queue>	CommandQueue;

	// The image format for all heightmaps
	// This has external linkage to some default params for functions in this 
	// namespace
	compute::image_format						ImageFormat = compute::image_format(CL_R, CL_UNSIGNED_INT16);

	static std::queue<std::function<void()>>	KernelQueue;
	static std::thread							KernelThread;
	static std::atomic<bool>					bKernelRunning;

	// Ensures that the device, context, queue are set up
	static void EnsureStateIsSetup()
	{
		if (Devices.size() == 0)
		{
			Devices.push_back(compute::system::default_device());
		}

		if (Context.get() == nullptr)
		{
			Context = unique_ptr<compute::context>(new compute::context(Devices));
		}

		if (CommandQueue.get() == nullptr)
		{
			CommandQueue = unique_ptr<compute::command_queue>(
				new compute::command_queue(*Context.get(), Devices[0]));
		}
	}

	void PushKernel(std::function<void()> KernelFunc)
	{
		KernelQueue.push(KernelFunc);
	}

	// Heightmap Ctor. Just allocates the image on the device side 
	Heightmap::Heightmap(int SizeX, int SizeY, boost::compute::image_format inImageFormat)
	{
		EnsureStateIsSetup();
		Image = compute::image2d(*Context.get(), SizeX, SizeY, inImageFormat);
	}

	Heightmap::operator TArray<uint16>() const
	{
		if (this->Image.format() != boost::compute::image_format(CL_R, CL_UNSIGNED_INT16))
			throw std::runtime_error("Wrong heightmap type conversion");

		TArray<uint16> OutArray;
		OutArray.SetNumUninitialized(Image.width() * Image.height());

		// Copy from the device to the host
		CommandQueue->enqueue_read_image(Image, Image.origin(), Image.size(), OutArray.GetData());

		return OutArray;
	}

	void* Heightmap::CreateRawCopy() const
	{
		uint8* OutData = new uint8[Image.get_memory_size()];

		CommandQueue->enqueue_read_image(Image, Image.origin(), Image.size(), OutData);

		return OutData;
	}

	Heightmap::operator TArray<float>() const
	{
		if (this->Image.format() != boost::compute::image_format(CL_RGBA, CL_FLOAT))
			throw std::runtime_error("Wrong heightmap type conversion");

		TArray<float> OutArray;
		OutArray.SetNumUninitialized(Image.width() * Image.height() * 4);

		// Copy from the device to the host
		CommandQueue->enqueue_read_image(Image, Image.origin(), Image.size(), OutArray.GetData());

		return OutArray;
	}

	void SetDevices(vector<compute::device> Devices)
	{
		LandscapeGeneration::Devices = Devices;
		LandscapeGeneration::Context = unique_ptr<compute::context>(
			new compute::context(LandscapeGeneration::Devices));
		LandscapeGeneration::CommandQueue = unique_ptr<compute::command_queue>(
			new compute::command_queue(*LandscapeGeneration::Context.get(), LandscapeGeneration::Devices[0]));
	}

	shared_ptr<Heightmap> CreateHeightmap(int SizeX, int SizeY, boost::compute::image_format inImageFormat)
	{
		return std::shared_ptr<Heightmap>(new Heightmap(SizeX, SizeY, inImageFormat));
	}

	void Tick()
	{
		if (!bKernelRunning)
		{
			if (KernelThread.joinable())
				KernelThread.join();

			if (KernelQueue.size() != 0)
			{
				bKernelRunning = true;

				// Create the new thread and make sure it sets bKernelRunning to false once it's done
				KernelThread = std::thread([]() -> void
				{
					KernelQueue.front()();
					KernelQueue.pop();
					bKernelRunning = false;
				});
			}
		}
	}

	namespace Kernels
	{
		static std::string const GetKernelsPath()
		{
			FString CLPath = FPaths::GameSourceDir() + "Forest/Kernels/";

			return std::string(TCHAR_TO_UTF8(*CLPath));
		}

		static std::string const GetBuildOptions()
		{
			return " -g -w -cl-kernel-arg-info";
		}

		void PerlinNoise(compute::image2d& Heightmap,
			float noiseSize, int32 seed, int32 depth, float amplitude)
		{
			using compute::dim;

			// Create the images needed for this kernel
			compute::image2d input_image(*Context.get(), Heightmap.width(), Heightmap.height(), ImageFormat);

			// build box filter program
			compute::program program =
				create_with_source_file({ GetKernelsPath() + "perlin.cl" }, *Context.get());

			program.build("-I \"" + GetKernelsPath() + "\"");

			// setup perlin kernel
			compute::kernel kernel(program, "perlin");
			kernel.set_arg(0, input_image);
			kernel.set_arg(1, Heightmap);
			kernel.set_arg(2, noiseSize);
			kernel.set_arg(3, seed);
			kernel.set_arg(4, depth);
			kernel.set_arg(5, amplitude);

			// execute the kernel
			CommandQueue->enqueue_nd_range_kernel(kernel, dim(0, 0), input_image.size(), dim(1, 1));
		}

		void WarpedPerlinNoise(compute::image2d& Heightmap,
			float noiseSize, int32 seed, int32 depth, float amplitude)
		{
			using compute::dim;

			// Create the images needed for this kernel
			compute::image2d input_image(*Context.get(), Heightmap.width(), Heightmap.height(), ImageFormat);

			// build box filter program
			compute::program program =
				create_with_source_file({ GetKernelsPath() + "perlin.cl", GetKernelsPath() + "warpedperlin.cl" }, *Context.get());

			program.build("-I \"" + GetKernelsPath() + "\"");

			// setup perlin kernel
			compute::kernel kernel(program, "warpedperlin");
			kernel.set_arg(0, input_image);
			kernel.set_arg(1, Heightmap);
			kernel.set_arg(2, noiseSize);
			kernel.set_arg(3, seed);
			kernel.set_arg(4, depth);
			kernel.set_arg(5, amplitude);

			// execute the kernel
			CommandQueue->enqueue_nd_range_kernel(kernel, dim(0, 0), input_image.size(), dim(1, 1));
		}

		void Mix(compute::image2d& LHeightMap,
			compute::image2d& RHeightMap,
			compute::image2d& OutputHeightmap,
			EMixType MixType)
		{
			// Make sure that the enum is the correct size and the images are the correct sizes
			static_assert(sizeof(MixType) == sizeof(cl_uchar), "sizeof(MixType) must equal sizeof(uchar)!");
			check(LHeightMap.width() == RHeightMap.width() && LHeightMap.height() == RHeightMap.height());
			check(RHeightMap.width() == OutputHeightmap.width() && RHeightMap.height() == OutputHeightmap.height());

			using compute::dim;

			compute::program program =
				create_with_source_file({ GetKernelsPath() + "mix.cl" }, *Context.get());

			program.build("-I \"" + GetKernelsPath() + "\"");

			// setup box filter kernel
			compute::kernel kernel(program, "mix_kernel");
			kernel.set_arg(0, OutputHeightmap);
			kernel.set_arg(1, LHeightMap);
			kernel.set_arg(2, RHeightMap);
			kernel.set_arg(3, (cl_uchar)MixType);

			// execute the box filter kernel
			CommandQueue->enqueue_nd_range_kernel(kernel, dim(0, 0), LHeightMap.size(), dim(1, 1));
		}

		void VoronoiNoise(compute::image2d& Heightmap,
			int32 noiseSize,
			int32 seed,
			float amplitude)
		{
			using compute::dim;

			// Create the images needed for this kernel
			compute::image2d input_image(*Context.get(), Heightmap.width(), Heightmap.height(), ImageFormat);

			// build box filter program
			compute::program program =
				create_with_source_file({ GetKernelsPath() + "perlin.cl", GetKernelsPath() + "voronoi.cl" }, *Context.get());

			program.build("-I \"" + GetKernelsPath() + "\"");

			// setup box filter kernel
			compute::kernel kernel(program, "voronoi");
			kernel.set_arg(0, input_image);
			kernel.set_arg(1, Heightmap);
			kernel.set_arg(2, noiseSize);
			kernel.set_arg(3, seed);
			kernel.set_arg(4, amplitude);

			// execute the box filter kernel
			CommandQueue->enqueue_nd_range_kernel(kernel, dim(0, 0), input_image.size(), dim(1, 1));
		}

		void Constant(compute::image2d& Heightmap,
			int32 height)
		{
			using compute::dim;

			auto size = Heightmap.width() * Heightmap.height();

			// Create an array to fill up the device memory with
			const std::unique_ptr<uint16[]> ConstantHeightArray(new uint16[size]);
			
			for (int i = 0; i < size; i++)
			{
				ConstantHeightArray[i] = (uint16)height;
			}

			CommandQueue->enqueue_write_image(Heightmap, Heightmap.origin(), Heightmap.size(), ConstantHeightArray.get());
		}

		void Erosion(boost::compute::image2d& Heightmap)
		{
			using compute::dim;

			const float DeltaTime = 0.1f;
			const auto FluxImageFormat = compute::image_format(CL_RGBA, CL_FLOAT);
			const auto WaterImageFormat = compute::image_format(CL_R, CL_FLOAT);

			auto waterHeight	= CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			auto hardness		= CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			auto sediment		= CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			auto sedimentCap	= CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			
			auto inFluxImage	= CreateHeightmap(Heightmap.width(), Heightmap.height(), FluxImageFormat);
			auto outFluxImage	= CreateHeightmap(Heightmap.width(), Heightmap.height(), FluxImageFormat);
			auto velocityImage	= CreateHeightmap(Heightmap.width(), Heightmap.height(), FluxImageFormat);

			{
				const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
					__kernel void hardness_const(
						__write_only image2d_t  outputImage
					)
				{
					const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
						CLK_ADDRESS_CLAMP_TO_EDGE |
						CLK_FILTER_NEAREST;

					// Store each work-item's unique row and column
					int x = get_global_id(0);
					int y = get_global_id(1);

					write_imagef(outputImage, (int2)(x, y), 1.f);
				}
				);

				compute::program hardness_program = boost::compute::program::create_with_source(source, *Context.get());

				hardness_program.build();
				compute::kernel hardness_const_kernel(hardness_program, "hardness_const");

				hardness_const_kernel.set_arg(0, hardness->Image);
				CommandQueue->enqueue_nd_range_kernel(hardness_const_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();
			}

			compute::program program =
				create_with_source_file({ GetKernelsPath() + "perlin.cl", GetKernelsPath() + "erosion.cl" }, *Context.get());

			((ue_compute_program*)(&program))->build("-I \"" + GetKernelsPath() + "\"");
			
			// Adds a random amount of rainfall
			compute::kernel rainfall_kernel(program, "rainfall");
			compute::kernel flux_kernel(program, "flux");
			compute::kernel k_factor_kernel(program, "calculate_k_factor");
			compute::kernel calculate_velocity_kernel(program, "calculate_velocity");
			compute::kernel calculate_sediment_capacity_kernel(program, "calculate_sediment_capacity");
			compute::kernel calculate_erosion_deposition_kernel(program, "calculate_erosion_deposition");
			
			for (int i = 0; i < 100; i++)
			{
				rainfall_kernel.set_args(
					waterHeight->Image,		// Water Height in
					waterHeight->Image,		// Water Height out
					(cl_uint)1000u + i,		// Seed
					(cl_float)DeltaTime,	// DeltaTime
					(cl_float)10.f			// WaterMul
				);

				CommandQueue->enqueue_nd_range_kernel(rainfall_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();
			}

			for (int i = 0; i < 50; i++)
			{
				// Calculate flux and ping-pong flux images
				{
					// Calculates the flux
					flux_kernel.set_args(
						Heightmap,				// Terrain Height in
						waterHeight->Image,		// Water Height in
						inFluxImage->Image,		// Flux in
						outFluxImage->Image,	// Flux out
						(cl_float)DeltaTime		// DeltaTime
					);

					CommandQueue->enqueue_nd_range_kernel(flux_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
					CommandQueue->finish();

					// Calculates the scaling factor for the flux and scales the flux
					k_factor_kernel.set_args(
						waterHeight->Image,		// Water Height in
						outFluxImage->Image,	// Flux in
						outFluxImage->Image,	// Flux out
						(cl_float)DeltaTime		// DeltaTime
					);

					CommandQueue->enqueue_nd_range_kernel(k_factor_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
					CommandQueue->finish();

					// Make sure to ping-pong after k factor
					std::swap(inFluxImage, outFluxImage);
				}

				// This doesn't have to be ping pongd
				compute::kernel calculate_water_height_kernel(program, "calculate_water_height_change");
				calculate_water_height_kernel.set_args(
					waterHeight->Image,		// Water Height in
					waterHeight->Image,		// Water Height out
					inFluxImage->Image,		// Flux in
					(cl_float) 0.1f			// DeltaTime
				);

				CommandQueue->enqueue_nd_range_kernel(calculate_water_height_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();

				calculate_velocity_kernel.set_args(
					inFluxImage->Image,		// Flux in
					velocityImage->Image,	// Velocity out
					(cl_float) 0.1f			// DeltaTime
				);

				CommandQueue->enqueue_nd_range_kernel(calculate_velocity_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();

				calculate_sediment_capacity_kernel.set_args(
					(cl_float) 1.f,			// Sediment capacity
					(cl_float) 1024.f,			// maxErosionDepth
					Heightmap,				// Terrain Height in
					waterHeight->Image,		// Water height in
					velocityImage->Image,	// Velocity in
					sedimentCap->Image		// Sediment Capacity Out
				);

				CommandQueue->enqueue_nd_range_kernel(calculate_sediment_capacity_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();

				calculate_erosion_deposition_kernel.set_args(
					Heightmap,				// Terrain Height in
					Heightmap,				// Terrain Height out
					hardness->Image,		// Terrain Hardness in
					sediment->Image,		// Sediment in
					sedimentCap->Image,		// Sediment capacity in

					(cl_float) 0.1f,		// deposition speed
					(cl_float) 1.f,			// sedimentCoefficient
					(cl_float) 0.01f,		// softeningCoefficient
					(cl_float) 0.1f,		// hardnessMin
					(cl_float)DeltaTime
				);

				CommandQueue->enqueue_nd_range_kernel(calculate_erosion_deposition_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();
			}

			//Heightmap = sedimentCap->Image;

			//compute::copy(
			//	hostBuffer.begin(), hostBuffer.end(), buffer.begin(), *CommandQueue.get()
			//);

			//FString Fs = FString(ANSI_TO_TCHAR(hostBuffer.data()));
			//UE_LOG(LogTemp, Warning, TEXT("%s"), *Fs);
			//UE_LOG(LogTemp, Warning, TEXT("asfkahfkld"));
		}
	}
}