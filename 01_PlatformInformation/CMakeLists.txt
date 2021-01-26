# 01 - Minimum CMake Version
# ############
cmake_minimum_required(VERSION 3.10)

# 2 - set the project name and version
# ############
project(PlatformInformation VERSION 1.0)

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


# 6 - add include folder to 
# ############
include_directories(${CMAKE_BINARY_DIR}/configuration)

# 7 - add the executable
# ############
file(GLOB SRC_FILES 	
		src/*.cpp
		src/*.h)

add_executable(PlatformInformation ${SRC_FILES})

# 8 - link libraries
# ############
target_link_libraries(PlatformInformation OpenCL::OpenCL)