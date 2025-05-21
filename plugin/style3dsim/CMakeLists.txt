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
		set(Style3DSimulatorDir ${CMAKE_CURRENT_SOURCE_DIR}/Style3DSimulatorBin/win)
	elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
		set(Style3DSimulatorDir ${CMAKE_CURRENT_SOURCE_DIR}/Style3DSimulatorBin/linux)
	else() 
		#TODO
	endif()
endif()

find_package(Style3DSimulator CONFIG REQUIRED PATHS ${Style3DSimulatorDir}/lib/cmake/Style3DSimulator)
target_link_libraries(${TARGET_NAME} PUBLIC Style3DSimulator)

# install lib
function(custom_install lib_path dest)
	install(FILES ${lib_path} DESTINATION ${dest})
	add_custom_command(TARGET ${TARGET_NAME} POST_BUILD COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${lib_path} $<TARGET_FILE_DIR:${TARGET_NAME}>)
endfunction()

if(${CMAKE_SYSTEM_NAME} STREQUAL "Windows")
	custom_install(${Style3DSimulatorDir}/bin/Style3DSimulator.dll ${CMAKE_INSTALL_BINDIR})
	custom_install(${Style3DSimulatorDir}/bin/Simulation.dll ${CMAKE_INSTALL_BINDIR})
	custom_install(${Style3DSimulatorDir}/bin/Core.dll ${CMAKE_INSTALL_BINDIR})
	custom_install(${Style3DSimulatorDir}/bin/CudaCore.dll ${CMAKE_INSTALL_BINDIR})
	custom_install(${Style3DSimulatorDir}/bin/PluginCpu.dll ${CMAKE_INSTALL_BINDIR})
	custom_install(${Style3DSimulatorDir}/bin/PluginCuda.dll ${CMAKE_INSTALL_BINDIR})
	custom_install(${Style3DSimulatorDir}/bin/tbb.dll ${CMAKE_INSTALL_BINDIR})
elseif(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
	custom_install(${Style3DSimulatorDir}/lib/libStyle3DSimulator.so ${CMAKE_INSTALL_LIBDIR})
	custom_install(${Style3DSimulatorDir}/lib/libSimulation.so ${CMAKE_INSTALL_LIBDIR})
	custom_install(${Style3DSimulatorDir}/lib/libCore.so ${CMAKE_INSTALL_LIBDIR})
	custom_install(${Style3DSimulatorDir}/lib/libCudaCore.so ${CMAKE_INSTALL_LIBDIR})
	custom_install(${Style3DSimulatorDir}/lib/libPluginCpu.so ${CMAKE_INSTALL_LIBDIR})
	custom_install(${Style3DSimulatorDir}/lib/libPluginCuda.so ${CMAKE_INSTALL_LIBDIR})
	custom_install(${Style3DSimulatorDir}/lib/libtbb.so.2 ${CMAKE_INSTALL_LIBDIR})
endif()
