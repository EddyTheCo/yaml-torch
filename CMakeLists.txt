cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(yaml-torch VERSION 0.1 DESCRIPTION "interface to torch from config files in YAML" LANGUAGES CXX)

find_package(Torch REQUIRED)

add_library(yaml-torch yaml-torch.cpp include/utils/yaml-torch.hpp)

target_link_libraries(yaml-torch PUBLIC  "${TORCH_LIBRARIES}")


if(NOT TARGET yaml-cpp)
	find_package(YAML-CPP CONFIG QUIET)
	if(NOT YAML-CPP_FOUND)
		message(STATUS "YAML-CPP was not found. Downloading")

		FetchContent_Declare(
			yaml-cpp
			GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
			GIT_TAG yaml-cpp-0.7.0
			)
		FetchContent_MakeAvailable(yaml-cpp)
		install(TARGETS yaml-cpp EXPORT yaml-torchTargets DESTINATION lib)
	else()
		message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
	endif(NOT YAML-CPP_FOUND)
endif(NOT TARGET yaml-cpp)

target_link_libraries(yaml-torch PUBLIC  yaml-cpp)
target_include_directories(yaml-torch PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(DIRECTORY include/ DESTINATION include/)

install(EXPORT yaml-torchTargets
	FILE yaml-torchTargets.cmake
	DESTINATION lib/cmake/yaml-torch
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/yaml-torchConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/yaml-torch"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/yaml-torchConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/yaml-torchConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/yaml-torchConfigVersion.cmake
	DESTINATION lib/cmake/yaml-torch
	)
export(EXPORT yaml-torchTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/yaml-torchTargets.cmake"
	)
