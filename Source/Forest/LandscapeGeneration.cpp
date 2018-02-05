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
#pragma warning(pop)

#include <memory>
#include <vector>
#include <queue>
#include <atomic>
#include <array>

using namespace std;
namespace compute = boost::compute;

// Same as the old program, except that it outputs the debug to UE_LOG
class ue_compute_program : public boost::compute::program
{
public:
	void build(const std::string &options = std::string())
	{
		const char *options_string = 0;

		if (!options.empty()) {
			options_string = options.c_str();
		}

		cl_int ret = clBuildProgram(get(), 0, 0, options_string, 0, 0);

#ifdef BOOST_COMPUTE_DEBUG_KERNEL_COMPILATION
		if (ret != CL_SUCCESS) {
			std::stringstream errorstringstream;
			// print the error, source code and build log
			errorstringstream << "Boost.Compute: "
				<< "kernel compilation failed (" << ret << ")\n"
				<< "--- source ---\n"
				<< source()
				<< "\n--- build log ---\n"
				<< build_log()
				<< std::endl;

			std::string errorstring(errorstringstream.str());

			UE_LOG(LogTemp, Warning, TEXT("%s"), ANSI_TO_TCHAR(errorstring.c_str()));
		}
#endif

		//if (ret != CL_SUCCESS) {
		//	BOOST_THROW_EXCEPTION(opencl_error(ret));
		//}
	}
};

namespace LandscapeGeneration
{
	static vector<compute::device>				Devices;
	static unique_ptr<compute::context>			Context;
	static unique_ptr<compute::command_queue>	CommandQueue;

	// The image format for all heightmaps
	static compute::image_format				ImageFormat = compute::image_format(CL_R, CL_UNSIGNED_INT16);

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
	Heightmap::Heightmap(int SizeX, int SizeY)
	{
		EnsureStateIsSetup();
		Image = compute::image2d(*Context.get(), SizeX, SizeY, ImageFormat);
	}

	Heightmap::operator TArray<uint16, FDefaultAllocator>() const
	{
		TArray<uint16> OutArray;
		OutArray.SetNumUninitialized(Image.width() * Image.height());

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

	shared_ptr<Heightmap> CreateHeightmap(int SizeX, int SizeY)
	{
		return std::shared_ptr<Heightmap>(new Heightmap(SizeX, SizeY));
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
				compute::program::create_with_source_file({ GetKernelsPath() + "perlin.cl" }, *Context.get());

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
				compute::program::create_with_source_file({ GetKernelsPath() + "perlin.cl", GetKernelsPath() + "warpedperlin.cl" }, *Context.get());

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
				compute::program::create_with_source_file({ GetKernelsPath() + "mix.cl" }, *Context.get());

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
				compute::program::create_with_source_file({ GetKernelsPath() + "perlin.cl", GetKernelsPath() + "voronoi.cl" }, *Context.get());

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

			auto size = Heightmap.width() * Heightmap.height();

			compute::program program =
				compute::program::create_with_source_file({ GetKernelsPath() + "perlin.cl", GetKernelsPath() + "erosion.cl" }, *Context.get());

			((ue_compute_program*)(&program))->build("-I \"" + GetKernelsPath() + "\"");
			
			compute::kernel kernel(program, "erosion");
			kernel.set_arg(0, Heightmap);
			kernel.set_arg(1, Heightmap);
			kernel.set_arg(2, 100u);
			kernel.set_arg(3, 1.f);
			kernel.set_arg(4, Heightmap);
			kernel.set_arg(5, Heightmap);

			CommandQueue->enqueue_nd_range_kernel(kernel, dim(0, 0), Heightmap.size(), dim(1, 1));
		}
	}
}