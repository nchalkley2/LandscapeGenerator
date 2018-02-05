// Copyright 1998-2017 Epic Games, Inc. All Rights Reserved.

#include "ForestGameMode.h"
#include "ForestHUD.h"
#include "ForestCharacter.h"
#include "UObject/ConstructorHelpers.h"

AForestGameMode::AForestGameMode()
	: Super()
{
	// set default pawn class to our Blueprinted character
	static ConstructorHelpers::FClassFinder<APawn> PlayerPawnClassFinder(TEXT("/Game/FirstPersonCPP/Blueprints/FirstPersonCharacter"));
	DefaultPawnClass = PlayerPawnClassFinder.Class;

	// use our custom HUD class
	HUDClass = AForestHUD::StaticClass();
}
