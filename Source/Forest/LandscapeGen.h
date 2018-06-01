// Fill out your copyright notice in the Description page of Project Settings.

#pragma once

#include "Core.h"
#include "Function.h"
#include "GameFramework/Actor.h"

#include "LandscapeGeneration.h"

#include <memory>
#include <future>

#include "LandscapeGen.generated.h"

class ALandscape;
class ALandscapeProxy;
class ALandscapeGen;

USTRUCT(BlueprintType, meta = (DisplayName = "Height Map"))
struct FHeightmapWrapper
{
	GENERATED_BODY()

	std::shared_ptr<LandscapeGeneration::Heightmap> Heightmap;
};

namespace LandscapeEditorUtils
{
	TMap<FIntPoint, uint16> GetHeightmapData(ALandscapeProxy* Landscape);
}

UCLASS()
class FOREST_API ALandscapeGen : public AActor
{
	GENERATED_UCLASS_BODY()

	//friend struct FHeightMapInfoWrapper;

public:
	UPROPERTY(EditAnywhere, Category="Landscape Actor")
		TWeakObjectPtr<ALandscape> Landscape;

	UFUNCTION(BlueprintCallable, Category = "Textures")
		UTexture2D* CreateTransientHeightmap();

	UFUNCTION(BlueprintCallable, Category = "Textures")
		UTexture2D* CreateTransientTexture();

	UFUNCTION(BlueprintCallable, Category = "Functions")
		void SetTransientHeightmap(UTexture2D* Texture, FHeightmapWrapper HeightMap);
	
	UFUNCTION(BlueprintPure, Category = "Noise")
		FHeightmapWrapper Perlin_Noise(float Size, int32 Seed, int32 Depth, float Amplitude);

	UFUNCTION(BlueprintPure, Category = "Noise")
		FHeightmapWrapper Warped_Perlin_Noise(float Size, int32 Seed, int32 Depth, float Amplitude);

	UFUNCTION(BlueprintPure, Category = "Noise")
		FHeightmapWrapper Voronoi_Noise(int32 Size, int32 Seed, float Amplitude);

	UFUNCTION(BlueprintPure, Category = "Functions")
		FHeightmapWrapper Erode_Landscape(FHeightmapWrapper HeightmapInput);

	UFUNCTION(BlueprintPure, Category = "Functions")
		FHeightmapWrapper Constant(int32 Height);

	UFUNCTION(BlueprintPure, Category = "Functions")
		FHeightmapWrapper Mix(FHeightmapWrapper LHeightMap, FHeightmapWrapper RHeightMap, EMixType MixType);

	UFUNCTION(BlueprintCallable, Category = "Functions")
		void SetHeightmap(FHeightmapWrapper HeightMap);

	// Tick in Editor
	virtual bool ShouldTickIfViewportsOnly() const override { return true; };

protected:
	// Called when the game starts or when spawned
	virtual void BeginPlay() override;

	TArray<uint16> GetLandscapeHeightmapSorted();

	std::future<TSharedPtr<SNotificationItem>> CreateNotification(const FText& InText);
	//void FinishNotification(FHeightMapInfoWrapper HeightInfo, const FText& InText, bool bFailure);

public:	
	// Called every frame
	virtual void Tick(float DeltaTime) override;
	
};
