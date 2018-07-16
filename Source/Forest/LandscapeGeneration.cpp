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
	compute::image_format						ImageFormat = compute::image_format(CL_R, CL_FLOAT);

	static std::queue<std::function<void()>>	KernelQueue;
	static std::thread							KernelThread;
	static std::atomic<bool>					bKernelRunning;

	// Ensures that the device, context, queue are set up
	static void EnsureStateIsSetup()
	{
		boost::compute::device NVIDIADevice;
		for (auto& Device : boost::compute::system::devices())
		{
			if (Device.vendor() == "NVIDIA Corporation")
			{
				NVIDIADevice = Device;
				break;
			}
		}

		if (Devices.size() == 0)
		{
			Devices.push_back(compute::system::default_device());
		}

		if (Context.get() == nullptr)
		{
			Context = unique_ptr<compute::context>(new compute::context(NVIDIADevice));
		}

		if (CommandQueue.get() == nullptr)
		{
			CommandQueue = unique_ptr<compute::command_queue>(
				new compute::command_queue(*Context.get(), NVIDIADevice));
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
		// Can't use a switch here because boost::compute::image_format is non const
		if (this->Image.format() != boost::compute::image_format(CL_R, CL_UNSIGNED_INT16)
		 && this->Image.format() != boost::compute::image_format(CL_R, CL_FLOAT))
			throw std::runtime_error("Wrong heightmap type conversion");

		if (this->Image.format() == boost::compute::image_format(CL_R, CL_UNSIGNED_INT16))
		{
			TArray<uint16> OutArray;
			OutArray.SetNumUninitialized(Image.width() * Image.height());

			// Copy from the device to the host
			CommandQueue->enqueue_read_image(Image, Image.origin(), Image.size(), OutArray.GetData());

			return OutArray;
		}

		if (this->Image.format() == boost::compute::image_format(CL_R, CL_FLOAT))
		{
			TArray<uint16> OutArray;
			OutArray.SetNumUninitialized(Image.width() * Image.height());

			float* RawCopy = (float*)this->CreateRawCopy();
			const auto PxNum = Image.width() * Image.height();

			for (int32 i = 0; i < PxNum; i++)
			{
				RawCopy[i] = FMath::Clamp(RawCopy[i], 0.f, (float)UINT16_MAX);

				OutArray[i] = (uint16)roundf(RawCopy[i]);
			}

			delete[] RawCopy;

			return OutArray;
		}

		throw std::runtime_error("Undefined control path. An exception should have been thrown at \"Wrong heightmap type conversion\".");
		return TArray<uint16>();
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
			float height)
		{
			using compute::dim;

			auto size = Heightmap.width() * Heightmap.height();

			// Create an array to fill up the device memory with
			const std::unique_ptr<float[]> ConstantHeightArray(new float[size]);
			
			for (int i = 0; i < size; i++)
			{
				ConstantHeightArray[i] = height;
			}

			CommandQueue->enqueue_write_image(Heightmap, Heightmap.origin(), Heightmap.size(), ConstantHeightArray.get());
		}

		ErosionParams Erosion(ErosionParams inputMaps,
			int32 iterations,
			float DeltaTime,
			float waterMul,
			float softeningCoefficient,
			float maxErosionDepth,
			float sedimentCapacity
			)
		{
			using compute::dim;

			const auto FluxImageFormat = compute::image_format(CL_RGBA, CL_FLOAT);
			const auto WaterImageFormat = compute::image_format(CL_R, CL_FLOAT);

			auto& Heightmap		= inputMaps.height->Image;
			auto waterHeight	= inputMaps.water;//CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			auto hardness		= inputMaps.hardness;//CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			auto sediment		= inputMaps.sediment;//CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			auto sediment2		= CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			auto sedimentCap	= inputMaps.sedimentCapacity;//CreateHeightmap(Heightmap.width(), Heightmap.height(), WaterImageFormat);
			
			auto inFluxImage	= inputMaps.flux;//CreateHeightmap(Heightmap.width(), Heightmap.height(), FluxImageFormat);
			auto outFluxImage	= CreateHeightmap(Heightmap.width(), Heightmap.height(), FluxImageFormat);
			auto velocityImage	= inputMaps.velocity;//CreateHeightmap(Heightmap.width(), Heightmap.height(), FluxImageFormat);

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

					write_imagef(outputImage, (int2)(x, y), 0.f);
				}
				);

				compute::program hardness_program = boost::compute::program::create_with_source(source, *Context.get());

				hardness_program.build();
				compute::kernel hardness_const_kernel(hardness_program, "hardness_const");

				hardness_const_kernel.set_arg(0, hardness->Image);
				CommandQueue->enqueue_nd_range_kernel(hardness_const_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();
			}

			/*{
				const char source[] = BOOST_COMPUTE_STRINGIZE_SOURCE(
					__kernel void rainfall_const(
						__write_only image2d_t  outputImage
					)
				{
					const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
						CLK_ADDRESS_CLAMP_TO_EDGE |
						CLK_FILTER_NEAREST;

					// Store each work-item's unique row and column
					int x = get_global_id(0);
					int y = get_global_id(1);

					write_imagef(outputImage, (int2)(x, y), 12.f);
				}
				);

				compute::program rainfall_program = boost::compute::program::create_with_source(source, *Context.get());

				rainfall_program.build();
				compute::kernel rainfall_const_kernel(rainfall_program, "rainfall_const");

				rainfall_const_kernel.set_arg(0, waterHeight->Image);
				CommandQueue->enqueue_nd_range_kernel(rainfall_const_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();
			}*/

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
			compute::kernel move_sediment_kernel(program, "move_sediment");
			
			for (int i = 0; i < iterations; i++)
			{
				rainfall_kernel.set_args(
					waterHeight->Image,		// Water Height in
					waterHeight->Image,		// Water Height out
					(cl_uint)1000u + i,		// Seed
					(cl_float)DeltaTime,	// DeltaTime
					(cl_float)waterMul		// WaterMul
				);

				CommandQueue->enqueue_nd_range_kernel(rainfall_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();

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
					(cl_float)DeltaTime		// DeltaTime
				);

				CommandQueue->enqueue_nd_range_kernel(calculate_water_height_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();

				calculate_velocity_kernel.set_args(
					inFluxImage->Image,		// Flux in
					velocityImage->Image,	// Velocity out
					(cl_float)DeltaTime		// DeltaTime
				);

				CommandQueue->enqueue_nd_range_kernel(calculate_velocity_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();

				calculate_sediment_capacity_kernel.set_args(
					(cl_float)sedimentCapacity,			// Sediment capacity
					(cl_float)maxErosionDepth,		// maxErosionDepth
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
					sediment2->Image,		// Sediment out
					sedimentCap->Image,		// Sediment capacity in
					waterHeight->Image,		// Water height in
					waterHeight->Image,		// Water height out

					(cl_float) 1.f,			// deposition speed
					(cl_float) 1.0f,		// sedimentCoefficient
					(cl_float)softeningCoefficient,		// softeningCoefficient
					(cl_float) 0.1f,		// hardnessMin
					(cl_float)DeltaTime
				);

				CommandQueue->enqueue_nd_range_kernel(calculate_erosion_deposition_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();

				/*move_sediment_kernel.set_args(
					sediment2->Image,		// Sediment in
					sediment->Image,		// Sediment out
					velocityImage->Image,	// Velocity in

					(cl_float)DeltaTime
				);

				CommandQueue->enqueue_nd_range_kernel(move_sediment_kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
				CommandQueue->finish();*/
			}

			//Heightmap = inFluxImage->Image;

			//Heightmap = sedimentCap->Image;

			//compute::copy(
			//	hostBuffer.begin(), hostBuffer.end(), buffer.begin(), *CommandQueue.get()
			//);

			//FString Fs = FString(ANSI_TO_TCHAR(hostBuffer.data()));
			//UE_LOG(LogTemp, Warning, TEXT("%s"), *Fs);
			//UE_LOG(LogTemp, Warning, TEXT("asfkahfkld"));

			return ErosionParams{waterHeight, hardness, sediment, sedimentCap, outFluxImage, velocityImage};
		}
	}
}