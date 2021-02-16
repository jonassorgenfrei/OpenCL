# 01 - Minimum CMake Version
# ############
cmake_minimum_required(VERSION 3.10)

# 2 - set the project name and version
# ############
project(MatrixMult VERSION 1.0)

# Output Dir (optional)
set(RuntimeOutputDir ${CMAKE_CURRENT_SOURCE_DIR}/build)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${RuntimeOutputDir})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${RuntimeOutputDir})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${RuntimeOutputDir})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${RuntimeOutputDir})

# 3 - specify the C++ standard
# ############
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# 4 - check for packages
# ############

# check for OpenCL
find_package( OpenCL REQUIRED )

# 5 - configure header file (config)
# ############
configure_file(configuration/config.h.in configuration/config.h)
# configure root directory to get relative references for files
configure_file(configuration/root_directory.h.in configuration/root_directory.h)
include_directories(${CMAKE_BINARY_DIR}/configuration)


# 6 - add include folder to 
# ############
include_directories(${CMAKE_BINARY_DIR}/configuration)



# 7 - add the executable
# ############
# find any cpp, h and hpp files
file(GLOB SRC_FILES 	
		src/*.cpp
		src/*.h
		src/*.hpp)

# add external files like ocl kernels
file(GLOB_RECURSE Kernels
	"kernel/*.cl"
)



# WINDOWS SYSTEM
if(WIN32)
	# dont build ZERO_CHECK
	set(CMAKE_SUPPRESS_REGENERATION true)
	# cmake Folder ALL_BUILD in Filter Subfolder
	set_property(GLOBAL PROPERTY USE_FOLDERS ON)
	
	# if Visual Studio
	if(MSVC)
		# ${PROJECT_NAME} as start Project
		set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT ${PROJECT_NAME})
	endif()
endif()



add_executable(${PROJECT_NAME} ${SRC_FILES} ${Kernels})

# erstellen der filter fuer die external-files
foreach(source IN LISTS Kernels)
    get_filename_component(source_path "${source}" PATH)
    file(RELATIVE_PATH pathR "${CMAKE_CURRENT_SOURCE_DIR}" "${source_path}")
    string(REPLACE "/" "\\" source_path_ide "${pathR}")
    source_group("${source_path_ide}" FILES "${source}")
endforeach()


# 8 - link libraries
# ############
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)