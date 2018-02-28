// Copyright 1998-2017 Epic Games, Inc. All Rights Reserved.

using UnrealBuildTool;
using System;
using System.IO;

public class Forest : ModuleRules
{
    private string ModulePath
    {
        get { return ModuleDirectory; }
    }

    private string ThirdPartyPath
    {
        get { return Path.GetFullPath(Path.Combine(ModulePath, "../../ThirdParty/")); }
    }

    public Forest(ReadOnlyTargetRules Target) : base(Target)
	{
        // Get the engine path. Ends with "Engine/"
        string EnginePath = Path.GetFullPath(BuildConfiguration.RelativeEnginePath);
        // Now get the base of UE4's modules dir (could also be Developer, Editor, ThirdParty)
        string EditorPath = EnginePath + "Source/Editor/";
        string RuntimePath = EnginePath + "Source/Runtime/";

        PCHUsage = PCHUsageMode.UseExplicitOrSharedPCHs;

        PrivateDependencyModuleNames.AddRange(new string[] {
                "LandscapeEditor",
                // Taken from LandscapeEditor
                "Core",
                "CoreUObject",
                "ApplicationCore",
                "Slate",
                "SlateCore",
                "EditorStyle",
                "Engine",
                "Landscape",
                "RenderCore",
                "RHI",
                "InputCore",
                "UnrealEd",
                "PropertyEditor",
                "ImageWrapper",
                "EditorWidgets",
                "Foliage",
                "ViewportInteraction",
                "VREditor"

                });
        PublicIncludePaths.AddRange(new string[] { EditorPath + "LandscapeEditor/Private" });

        PublicDependencyModuleNames.AddRange(new string[] { "Core", "CoreUObject", "Engine", "InputCore", "HeadMountedDisplay", "Landscape", "LandscapeEditor" });

        AddComputePath(Target);

    }

    public void AddComputePath(ReadOnlyTargetRules Target)
    {
        bUseRTTI = true;

        // Windows paths
        if (Target.Platform == UnrealTargetPlatform.Win32 || Target.Platform == UnrealTargetPlatform.Win64)
        {
            string OpenCLPath = "";
            string OpenCLIncludePath = "";
            string OpenCLLibPath = "";

            // OpenCL paths for NVIDIA GPU Computing Toolkit
            if (Environment.GetEnvironmentVariable("CUDA_PATH") != null)
            {
                OpenCLPath = Environment.GetEnvironmentVariable("CUDA_PATH");
                OpenCLIncludePath = Path.Combine(OpenCLPath, "include/");
                OpenCLLibPath = Path.Combine(OpenCLPath, "lib/x64/");
            }

            PublicDelayLoadDLLs.Add("OpenCL.dll");
            PublicIncludePaths.Add(OpenCLIncludePath);
            PublicAdditionalLibraries.Add(Path.Combine(OpenCLLibPath, "OpenCL.lib"));

            // Boost paths
            if (Environment.GetEnvironmentVariable("BOOST_ROOT") != null)
            {
                string BoostRoot = Environment.GetEnvironmentVariable("BOOST_ROOT");

                if (Target.WindowsPlatform.Compiler == WindowsCompiler.VisualStudio2017)
                {
                    string BoostLibDir = Path.Combine(BoostRoot, "lib64-msvc-14.1/");

                    PublicIncludePaths.Add(BoostRoot);
                    PublicLibraryPaths.Add(BoostLibDir);
                    PublicAdditionalLibraries.Add(Path.Combine(BoostLibDir, "libboost_chrono-vc141-mt-1_65_1.lib"));
                }
                else if (Target.WindowsPlatform.Compiler == WindowsCompiler.VisualStudio2015)
                {
                    string BoostLibDir = Path.Combine(BoostRoot, "lib64-msvc-14.0/");

                    PublicIncludePaths.Add(BoostRoot);
                    PublicLibraryPaths.Add(BoostLibDir);
                    PublicAdditionalLibraries.Add(Path.Combine(BoostLibDir, "libboost_chrono-vc140-mt-1_65_1.lib"));
                }
            }
        }
    }
}
