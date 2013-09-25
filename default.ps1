properties { 
  $base_dir = resolve-path .
  $build_dir = "$base_dir\build"
  $packageinfo_dir = "$base_dir\nuspecs"
  $release_dir = "$base_dir\Release"
  $sln_file = "$base_dir\GpuStructs.sln"
  $tools_dir = "$base_dir\tools"
  $version = "1.0.0"
  $config_cpu = "Release.cpu"
  $config_cu = "Release.cu"
  $run_tests = $true
}
Framework "4.0"
	
task default -depends Package

task Clean {
	remove-item -force -recurse $build_dir -ErrorAction SilentlyContinue
	remove-item -force -recurse $release_dir -ErrorAction SilentlyContinue
}

task Init -depends Clean {
	new-item $build_dir -itemType directory 
	new-item $release_dir -itemType directory 
}

task Compile -depends Init {
	msbuild $sln_file /p:"OutDir=$build_dir\cpu\;IntDir=$build_dir\cpu\obj\;Configuration=$config_cpu;LC=cpu;LD=" /m
	msbuild $sln_file /p:"OutDir=$build_dir\cpuV\;IntDir=$build_dir\cpuV\obj\;Configuration=$config_cpu;LC=cpu;LD=V" /m
	msbuild $sln_file /p:"OutDir=$build_dir\cu11\;IntDir=$build_dir\cu11\obj\;Configuration=$config_cu;LC=11;LD=" /m
	msbuild $sln_file /p:"OutDir=$build_dir\cu11V\;IntDir=$build_dir\cu11V\obj\;Configuration=$config_cu;LC=11;LD=V" /m
	msbuild $sln_file /p:"OutDir=$build_dir\cu20\;IntDir=$build_dir\cu20\obj\;Configuration=$config_cu;LC=20;LD=" /m
	msbuild $sln_file /p:"OutDir=$build_dir\cu20V\;IntDir=$build_dir\cu20V\obj\;Configuration=$config_cu;LC=20;LD=V" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\cu30\;IntDir=$build_dir\cu30\obj\;Configuration=$config_cu;LC=30;LD=" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\cu30V\;IntDir=$build_dir\cu30V\obj\;Configuration=$config_cu;LC=30;LD=V" /m
	#msbuild $sln_file /p:"OutDir=$build_dir\cu35\;IntDir=$build_dir\cu35\obj\;Configuration=$config_cu;LC=35;LD=" /m
	#msbuild $sln_file /target:Rebuild /p:"OutDir=$build_dir\cu35V\;IntDir=$build_dir\cu35V\obj\;Configuration=$config_cu;LC=35;LD=V" /m
}

task Test -depends Compile -precondition { return $run_tests } {
	$old = pwd
	cd $build_dir
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu11\Runtime.11.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu11V\Runtime.11V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu20\Runtime.20.Tests.dll" /noshadow
	& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu20V\Runtime.20V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu30\Runtime.30.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu30V\Runtime.30V.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu35\Runtime.35.Tests.dll" /noshadow
	#& $tools_dir\xUnit\xunit.console.clr4.exe "$build_dir\cu35V\Runtime.35V.Tests.dll" /noshadow
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

task Package -depends Dependency, Compile, Test {
	$spec_files = @(Get-ChildItem $packageinfo_dir)
	foreach ($spec in $spec_files)
	{
		& $tools_dir\NuGet.exe pack $spec.FullName -o $release_dir -Version $version -Symbols -BasePath $base_dir
	}
}
