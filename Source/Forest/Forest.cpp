// Copyright 1998-2017 Epic Games, Inc. All Rights Reserved.

#include "Forest.h"
#include "Modules/ModuleManager.h"
#include "EngineUtils.h"

class FForest : public FDefaultGameModuleImpl
{
	void* DLLHandle;

	virtual void StartupModule() override
	{
		DLLHandle = FPlatformProcess::GetDllHandle(TEXT("OpenCL.dll"));

		if (DLLHandle)
		{
			UE_LOG(LogTemp, Display, TEXT("Successfully loaded OpenCL.dll"));
			//GEngine->AddOnScreenDebugMessage(-1, 15.0f, FColor::Red, TEXT("Successfully loaded OpenCL.dll"));
		}
	}

	virtual void ShutdownModule() override
	{
		FPlatformProcess::FreeDllHandle(DLLHandle);
	}
};


IMPLEMENT_PRIMARY_GAME_MODULE(FForest, Forest, "Forest" );
 