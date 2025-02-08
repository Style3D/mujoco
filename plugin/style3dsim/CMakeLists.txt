# Copyright Linctex Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.21)

set(TARGET_NAME style3dsim)

set(MUJOCO_STYLE3DSIM_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../..
                              ${CMAKE_CURRENT_SOURCE_DIR}/../../src
)

set(MUJOCO_STYLE3DSIM_SRCS
	style3dsim.cc
    style3dsim.h
    register.cc
)

add_library(${TARGET_NAME} SHARED)
target_sources(${TARGET_NAME} PRIVATE ${MUJOCO_STYLE3DSIM_SRCS})
target_include_directories(${TARGET_NAME} PRIVATE ${MUJOCO_STYLE3DSIM_INCLUDE})
target_link_libraries(${TARGET_NAME} PRIVATE mujoco)
target_compile_options(
  ${TARGET_NAME}
  PRIVATE ${AVX_COMPILE_OPTIONS}
          ${MUJOCO_MACOS_COMPILE_OPTIONS}
          ${EXTRA_COMPILE_OPTIONS}
          ${MUJOCO_CXX_FLAGS}
)
target_link_options(
  ${TARGET_NAME}
  PRIVATE
  ${MUJOCO_MACOS_LINK_OPTIONS}
  ${EXTRA_LINK_OPTIONS}
)

# Link Style3DSimulator
if(NOT DEFINED Style3DSimulatorDir)
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
		set(Style3DSimulatorDir ${CMAKE_CURRENT_SOURCE_DIR}/Style3DSimulatorBin/)
	else() ##Darwin
		#set(Style3DSimulatorDir ${CMAKE_CURRENT_SOURCE_DIR}/Style3DSimulatorBin/Mac)
	endif()
endif()

find_package(Style3DSimulator CONFIG REQUIRED PATHS ${Style3DSimulatorDir}/lib/cmake/Style3DSimulator)
target_link_libraries(${TARGET_NAME} PUBLIC Style3DSimulator)

# install lib
function(custom_install simLib)
	if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
		install(FILES ${Style3DSimulatorDir}/bin/${simLib}.dll DESTINATION bin)
		add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${Style3DSimulatorDir}/bin/${simLib}.dll $<TARGET_FILE_DIR:${TARGET_NAME}>)
	endif()
endfunction()

custom_install(Style3DSimulator)
custom_install(Simulation)
custom_install(Core)
custom_install(CudaCore)
custom_install(PluginCpu)
custom_install(PluginCuda)
custom_install(tbb)
