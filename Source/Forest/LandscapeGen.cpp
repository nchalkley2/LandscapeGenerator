// Fill out your copyright notice in the Description page of Project Settings.

#include "LandscapeGen.h"
#include "EngineUtils.h"
#include "Classes/Landscape.h"
#include "Classes/LandscapeComponent.h"
#include "Classes/LandscapeInfo.h"
#include "Classes/LandscapeProxy.h"
#include "Engine/Texture2D.h"

// Disable warning for WITH_KISSFFT not being defined
#pragma warning(push)
#pragma warning(disable: 4668)
#include "LandscapeEdModeTools.h"
#include "LandscapeEditorUtils.h"
#pragma warning(pop)

#include "NotificationManager.h"
#include "SNotificationList.h"

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

#include <sstream>

#define LOCTEXT_NAMESPACE "Landscape Generation"

// Return the heightmap for the landscape
TMap<FIntPoint, uint16> LandscapeEditorUtils::GetHeightmapData(ALandscapeProxy* Landscape)
{
	TMap<FIntPoint, uint16> Data;
	FIntRect LandscapeBounds = Landscape->GetBoundingRect();

	FHeightmapAccessor<false> HeightmapAccessor(Landscape->GetLandscapeInfo());

	int32 min = 0;
	HeightmapAccessor.GetData(min, min, LandscapeBounds.Max.X, LandscapeBounds.Max.Y, Data);

	return Data;
}


#include <iostream>

namespace compute = boost::compute;

// Returns true on no error, false on error
static bool catch_error(std::function<void()> openclfunc)
{
	try
	{
		openclfunc();
	}
	catch (compute::opencl_error& e)
	{
		UE_LOG(LogTemp, Warning, TEXT("OpenCL Error: %s"), ANSI_TO_TCHAR(e.what()));
		return false;
	}
	catch (std::exception& e)
	{
		UE_LOG(LogTemp, Warning, TEXT("OpenCL Error: %s"), ANSI_TO_TCHAR(e.what()));
		return false;
	}

	return true;
}

// Sets default values
ALandscapeGen::ALandscapeGen(const FObjectInitializer& ObjectInitializer)
	: Super(ObjectInitializer)
{
	// Set this actor to call Tick() every frame.  You can turn this off to improve performance if you don't need it.
	PrimaryActorTick.bCanEverTick = true;
	PrimaryActorTick.bStartWithTickEnabled = true;
	PrimaryActorTick.bTickEvenWhenPaused = true;
}

TArray<uint16> ALandscapeGen::GetLandscapeHeightmapSorted()
{
	TArray<uint16> Data;
	static bool GotLandscapeStuff = false;
	if (!GotLandscapeStuff)
	{
		ULandscapeInfo::RecreateLandscapeInfo(GetWorld(), true);
		GotLandscapeStuff = true;
	}
	
	if (Landscape.IsValid())
	{
		auto LandscapeRef = Landscape.Get();

		// This heightmap isnt sorted, pixels will be randomly in the array
		auto HeightMap = LandscapeEditorUtils::GetHeightmapData(LandscapeRef);
		FIntRect LandscapeBounds = LandscapeRef->GetBoundingRect();

		// This is the output array
		TArray<uint16> Data;
		Data.Reserve(HeightMap.Num());

		// Load the heightmap in to the array
		int32 i = 0;
		for (int32 y = 0; y <= LandscapeBounds.Max.Y; y++)
		{
			for (int32 x = 0; x <= LandscapeBounds.Max.X; x++)
			{
				Data.Insert(*(HeightMap.Find(FIntPoint(x, y))), i);
				i++;
			}
		}

		return Data;
	}

	// Should probably throw an error here...
	return Data;
}

/*
void ALandscapeGen::CreateNotification(FHeightmapWrapper HeightInfo, const FText& InText)
{
	AsyncTask(ENamedThreads::GameThread, [=]()
	{
		FNotificationInfo Info(InText);
		Info.FadeInDuration = 0.1f;
		Info.FadeOutDuration = 5.0f;
		Info.ExpireDuration = 5.5f;
		Info.bUseThrobber = true;
		Info.bUseSuccessFailIcons = true;
		Info.bUseLargeFont = true;
		Info.bFireAndForget = false;
		Info.bAllowThrottleWhenFrameRateIsLow = false;
		auto NotificationItem = FSlateNotificationManager::Get().AddNotification(Info);

		NotificationItem->SetCompletionState(SNotificationItem::CS_Pending);

		HeightInfo->Notifications.Add(NotificationItem);
	});
}

void ALandscapeGen::FinishNotification(FHeightmapWrapper HeightInfo, const FText& InText, bool bFailure)
{
	AsyncTask(ENamedThreads::GameThread, [&]()
	{
		auto LastNotification = HeightInfo->Notifications.Pop();
		LastNotification->SetCompletionState(bFailure ? SNotificationItem::CS_Fail : SNotificationItem::CS_Success);
		LastNotification->ExpireAndFadeout();

		FNotificationInfo Info(InText);
		Info.FadeInDuration = 0.1f;
		Info.FadeOutDuration = 0.5f;
		Info.ExpireDuration = 1.5f;
		Info.bUseThrobber = false;
		Info.bUseSuccessFailIcons = true;
		Info.bUseLargeFont = true;
		Info.bFireAndForget = false;
		Info.bAllowThrottleWhenFrameRateIsLow = false;
		auto NotificationItem = FSlateNotificationManager::Get().AddNotification(Info);

		NotificationItem->SetCompletionState(bFailure ? SNotificationItem::CS_Fail : SNotificationItem::CS_Success);
		NotificationItem->ExpireAndFadeout();
	});
}
*/

FHeightmapWrapper ALandscapeGen::Constant(int32 Height)
{
	UE_LOG(LogTemp, Warning, TEXT("Constant"));
	FHeightmapWrapper NewHeightmap;

	if (Landscape.IsValid())
	{
		auto LandscapeRef = Landscape.Get();
		auto LandscapeBounds = LandscapeRef->GetBoundingRect();

		NewHeightmap.Heightmap = LandscapeGeneration::CreateHeightmap(
			LandscapeBounds.Max.X + 1, LandscapeBounds.Max.Y + 1);

		LandscapeGeneration::PushKernel([=]() -> void
		{
			LandscapeGeneration::Kernels::Constant(NewHeightmap.Heightmap->Image, Height);
		});
	}

	return NewHeightmap;
}

FHeightmapWrapper ALandscapeGen::Perlin_Noise(float Size, int32 Seed, int32 Depth, float Amplitude)
{
	UE_LOG(LogTemp, Warning, TEXT("Perlin Noise"));
	FHeightmapWrapper NewHeightmap;

	if (Landscape.IsValid())
	{
		auto LandscapeRef = Landscape.Get();
		auto LandscapeBounds = LandscapeRef->GetBoundingRect();

		NewHeightmap.Heightmap = LandscapeGeneration::CreateHeightmap(
			LandscapeBounds.Max.X + 1, LandscapeBounds.Max.Y + 1);

		LandscapeGeneration::PushKernel([=]() -> void
		{
			//CreateNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Generating Perlin Noise..."));

			catch_error([=]() -> void
			{
				LandscapeGeneration::Kernels::PerlinNoise(NewHeightmap.Heightmap->Image, Size, Seed, Depth, Amplitude);
			}); //?
				// Notify the user that a kernel finished
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Finished Generating Perlin Noise"), false) :
				// Notify the user that a kernel failed
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Failed Generating Perlin Noise"), true);
		});
	}

	return NewHeightmap;
}

FHeightmapWrapper ALandscapeGen::Warped_Perlin_Noise(float Size, int32 Seed, int32 Depth, float Amplitude)
{
	UE_LOG(LogTemp, Warning, TEXT("Warped Perlin Noise"));
	FHeightmapWrapper NewHeightmap;

	if (Landscape.IsValid())
	{
		auto LandscapeRef = Landscape.Get();
		auto LandscapeBounds = LandscapeRef->GetBoundingRect();

		NewHeightmap.Heightmap = LandscapeGeneration::CreateHeightmap(
			LandscapeBounds.Max.X + 1, LandscapeBounds.Max.Y + 1);

		LandscapeGeneration::PushKernel([=]() -> void
		{
			//CreateNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Generating Perlin Noise..."));

			catch_error([=]() -> void
			{
				LandscapeGeneration::Kernels::WarpedPerlinNoise(NewHeightmap.Heightmap->Image, Size, Seed, Depth, Amplitude);
			}); //?
				// Notify the user that a kernel finished
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Finished Generating Perlin Noise"), false) :
				// Notify the user that a kernel failed
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Failed Generating Perlin Noise"), true);
		});
	}

	return NewHeightmap;
}

FHeightmapWrapper ALandscapeGen::Voronoi_Noise(int32 Size, int32 Seed, float Amplitude)
{
	UE_LOG(LogTemp, Warning, TEXT("Voronoi Noise"));
	FHeightmapWrapper NewHeightmap;

	if (Landscape.IsValid())
	{
		auto LandscapeRef = Landscape.Get();
		auto LandscapeBounds = LandscapeRef->GetBoundingRect();

		NewHeightmap.Heightmap = LandscapeGeneration::CreateHeightmap(
			LandscapeBounds.Max.X + 1, LandscapeBounds.Max.Y + 1);

		LandscapeGeneration::PushKernel([=]() -> void
		{
			//CreateNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Generating Perlin Noise..."));

			catch_error([=]() -> void
			{
				LandscapeGeneration::Kernels::VoronoiNoise(NewHeightmap.Heightmap->Image, Size, Seed, Amplitude);
			}); //?
				// Notify the user that a kernel finished
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Finished Generating Perlin Noise"), false) :
				// Notify the user that a kernel failed
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Failed Generating Perlin Noise"), true);
		});
	}

	return NewHeightmap;
}

FHeightmapWrapper ALandscapeGen::Erode_Landscape(FHeightmapWrapper HeightmapInput)
{
	UE_LOG(LogTemp, Warning, TEXT("Erosion"));

	LandscapeGeneration::PushKernel([=]() -> void
	{
		//CreateNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Generating Perlin Noise..."));

		catch_error([=]() -> void
		{
			LandscapeGeneration::Kernels::Erosion(HeightmapInput.Heightmap->Image);
		}); //?
			// Notify the user that a kernel finished
			//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Finished Generating Perlin Noise"), false) :
			// Notify the user that a kernel failed
			//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Failed Generating Perlin Noise"), true);
	});

	return HeightmapInput;
}

FHeightmapWrapper ALandscapeGen::Mix(FHeightmapWrapper LHeightMap, FHeightmapWrapper RHeightMap, EMixType MixType)
{
	UE_LOG(LogTemp, Warning, TEXT("Mix"));
	FHeightmapWrapper NewHeightmap;

	// Check that the pointers are not null
	if (LHeightMap.Heightmap != nullptr && RHeightMap.Heightmap != nullptr)
	{
		NewHeightmap.Heightmap = LandscapeGeneration::CreateHeightmap(
			LHeightMap.Heightmap->Image.width(), LHeightMap.Heightmap->Image.height());

		LandscapeGeneration::PushKernel([=]() -> void
		{
			//CreateNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Mixing Heightmaps..."));

			catch_error([=]() -> void
			{
				LandscapeGeneration::Kernels::Mix(LHeightMap.Heightmap->Image,
					RHeightMap.Heightmap->Image, NewHeightmap.Heightmap->Image, MixType);
			}); //?
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Finished Mixing Heightmaps"), false) :
				// Notify the user that a kernel failed
				//FinishNotification(OutFuture, LOCTEXT("LandscapeGenNotifications", "Failed Mixing Heightmaps"), true);
		});
	}

	return NewHeightmap;
}

void ALandscapeGen::SetHeightmap(FHeightmapWrapper HeightMap)
{
	UE_LOG(LogTemp, Warning, TEXT("Set Heightmap"));

	// Check that the heightmap exists
	if (HeightMap.Heightmap != nullptr)
	{
		// This has to be added to the queue so that its executed in order
		LandscapeGeneration::PushKernel([=, this]() -> void
		{
			// Read from the device to this height map array
			TArray<uint16> HeightMapArray = *HeightMap.Heightmap.get();

			// We can't call the editorutil function from the async thread, so it has to be from here
			AsyncTask(ENamedThreads::GameThread, [=, this]()
			{
				UE_LOG(LogTemp, Warning, TEXT("Set Heightmap Async"));
				LandscapeEditorUtils::SetHeightmapData(Landscape.Get(), HeightMapArray);
			});
		});
	}
}

// Called when the game starts or when spawned
void ALandscapeGen::BeginPlay()
{
	Super::BeginPlay();
}

// Called every frame
void ALandscapeGen::Tick(float DeltaTime)
{
	Super::Tick(DeltaTime);

	LandscapeGeneration::Tick();
}

#undef LOCTEXT_NAMESPACE