cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(yaml-torch VERSION 0.1 DESCRIPTION "interface to torch from config files in YAML" LANGUAGES CXX)

find_package(Torch REQUIRED)
find_package(YAML-CPP REQUIRED)

add_library(yaml-torch yaml-torch.cpp include/utils/yaml-torch.hpp)

target_link_libraries(yaml-torch PUBLIC  "${TORCH_LIBRARIES}")

target_link_libraries(yaml-torch PUBLIC  yaml-cpp)

target_include_directories(yaml-torch PUBLIC  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS yaml-torch EXPORT yaml-torchTargets DESTINATION lib)
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
execute_process(COMMAND "git" "log" "--format='%H'" "-n 1"
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE  commit)
execute_process(COMMAND "git" "rev-parse" "--abbrev-ref" "HEAD" 
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        OUTPUT_VARIABLE  branch)

FILE(WRITE ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  ${PROJECT_NAME}:\n)
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t commit=${commit})
FILE(APPEND ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt  \t branch=${branch}\n)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/.${PROJECT_NAME}.txt DESTINATION .conf)
