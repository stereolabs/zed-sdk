cmake_minimum_required( VERSION 3.8.0 )

project(ZED_Positional_Tracking CSharp)

add_executable(${PROJECT_NAME}
  	Program.cs
	MainWindow.cs
	GLViewer.cs
	ZEDModel.cs
	App.config
	packages.config
	Properties/AssemblyInfo.cs
)
 
# Set the target platform to x64, since ZED SDK does not support 32-bits arch
target_compile_options(${PROJECT_NAME} PRIVATE "/platform:x64" )

# Set the .NET Framework version for the target.
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_TARGET_FRAMEWORK_VERSION "v4.6.1")

# Set the C# language version, otherwise default 3.0 is taken
set(CMAKE_CSharp_FLAGS "/langversion:7")

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DOTNET_REFERENCES
	"Microsoft.CSharp"
	"PresentationCore"
	"PresentationFramework"
	"System"
	"System.Xaml"
	"System.Data"
	"System.Linq"
	"System.Windows"
	"System.Windows.Forms"
	"System.Numerics"
	"System.Drawing"
	"WindowsBase"
)

set_property(TARGET ${PROJECT_NAME} PROPERTY VS_PACKAGE_REFERENCES 
	"OpenGL.Net_0.8.4"
	"OpenGL.Net.CoreUI_0.8.4"
	"Stereolabs.zed_3.3.1-rc"
	"sl_zed_interface_3.3.1-rc"
) 