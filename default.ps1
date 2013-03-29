properties { 
  $base_dir = resolve-path .
  $build_dir = "$base_dir\build"
  $packageinfo_dir = "$base_dir\nuspecs"
  $11_build_dir = "$build_dir\3.5\"
  $20_build_dir = "$build_dir\4.0\"
  $release_dir = "$base_dir\Release"
  $sln_file = "$base_dir\GpuStructs.sln"
  $tools_dir = "$base_dir\tools"
  $version = "1.0.0" #Get-Version-From-Git-Tag
  $11_config = "Release"
  $20_config = "Release.2"
  $run_tests = $true
}
Framework "4.0"

#include .\psake_ext.ps1
	
task default -depends Release

task Clean {
	remove-item -force -recurse $build_dir -ErrorAction SilentlyContinue
	remove-item -force -recurse $release_dir -ErrorAction SilentlyContinue
}

task Init -depends Clean {
	new-item $build_dir -itemType directory 
	new-item $release_dir -itemType directory 
}

task Compile -depends Init {
	msbuild $sln_file /p:"OutDir=$build_dir\11;Configuration=$11_config"
	msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\20;Configuration=$20_config"
}

task Test -depends Compile -precondition { return $run_tests } {
	$old = pwd
	cd $build_dir
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\11\System.WebEx.Tests.dll" /noshadow
	cd $old
}

task Dependency {
	$package_files = @(Get-ChildItem src -include *packages.config -recurse)
	foreach ($package in $package_files)
	{
		Write-Host $package.FullName
		& $tools_dir\NuGet.exe install $package.FullName -o packages
	}
}

task Release -depends Dependency, Compile, Test {
	cd $build_dir
	& $tools_dir\7za.exe a $release_dir\BclEx-Web.zip `
		*\System.WebEx.dll `
		*\System.WebEx.xml `
    	..\license.txt
	if ($lastExitCode -ne 0) {
		throw "Error: Failed to execute ZIP command"
    }
}

task Package -depends Release {
	$spec_files = @(Get-ChildItem $packageinfo_dir)
	foreach ($spec in $spec_files)
	{
		& $tools_dir\NuGet.exe pack $spec.FullName -o $release_dir -Version $version -Symbols -BasePath $base_dir
	}
}
