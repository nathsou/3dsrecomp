cmake_minimum_required(VERSION 3.20)
project(3dsrecomp C CXX OBJCXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(GAME sm3dl CACHE STRING "generated/<GAME> to link")
set(GEN_OPT -O2 CACHE STRING "optimisation of the generated code")

file(GLOB GEN_SRC CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/generated/${GAME}/*.c)
add_library(gamecode STATIC ${GEN_SRC})
target_include_directories(gamecode PRIVATE runtime generated/${GAME})
target_compile_options(gamecode PRIVATE ${GEN_OPT} -w -ffp-contract=off -fno-strict-aliasing -std=gnu11)

file(GLOB RT_SRC CONFIGURE_DEPENDS runtime/*.cpp runtime/*.mm runtime/services/*.cpp runtime/gpu/*.cpp runtime/gpu/*.mm runtime/audio/*.cpp)
add_executable(${GAME} ${RT_SRC})
target_include_directories(${GAME} PRIVATE runtime)
target_compile_options(${GAME} PRIVATE -O2 -g -Wall -Wno-unused-function -Wno-unused-variable)
find_package(SDL3 REQUIRED CONFIG)
target_link_libraries(${GAME} PRIVATE gamecode SDL3::SDL3 "-framework Metal" "-framework Foundation" "-framework AppKit" "-framework CoreText" "-Wl,-export_dynamic")
target_compile_definitions(${GAME} PRIVATE RECOMP_SOURCE_DIR="${CMAKE_SOURCE_DIR}" RECOMP_GAME="${GAME}")
set_source_files_properties(${RT_SRC} PROPERTIES COMPILE_OPTIONS "")
foreach(f ${RT_SRC})
  if(f MATCHES "\\.mm$")
    set_source_files_properties(${f} PROPERTIES COMPILE_OPTIONS "-fobjc-arc")
  endif()
endforeach()

# Fragment shaders of the shipped pipeline lists (games/<title id>/pipelines.bin) and the vertex program
# kernels (games/<title id>/vsprograms.bin), compiled ahead of time into build/shaders.metallib so the game
# never waits for the Metal compiler (see runtime/gpu/metal.mm). Only this game's lists when
# data/<GAME>/info.json names its title id.
set(GAME_TID "*")
if(EXISTS ${CMAKE_SOURCE_DIR}/data/${GAME}/info.json)
  file(READ ${CMAKE_SOURCE_DIR}/data/${GAME}/info.json GAME_INFO)
  string(JSON GAME_TID ERROR_VARIABLE GAME_INFO_ERR GET ${GAME_INFO} program_id)
  if(GAME_INFO_ERR)
    set(GAME_TID "*")
  endif()
endif()
file(GLOB PIPELINE_LISTS CONFIGURE_DEPENDS ${CMAKE_SOURCE_DIR}/games/${GAME_TID}/pipelines.bin
     ${CMAKE_SOURCE_DIR}/games/${GAME_TID}/vsprograms.bin)
execute_process(COMMAND xcrun -f metal RESULT_VARIABLE METAL_RC OUTPUT_QUIET ERROR_QUIET)
if(PIPELINE_LISTS AND METAL_RC EQUAL 0)
  add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/shaders.metallib
    COMMAND ${GAME} --emit-shaders ${CMAKE_BINARY_DIR}/shaders ${PIPELINE_LISTS}
    COMMAND sh ${CMAKE_SOURCE_DIR}/tools/build_metallib.sh ${CMAKE_BINARY_DIR}/shaders ${CMAKE_BINARY_DIR}/shaders.metallib
    DEPENDS ${GAME} ${PIPELINE_LISTS} ${CMAKE_SOURCE_DIR}/tools/build_metallib.sh
    COMMENT "Precompiling shaders")
  add_custom_target(shaders ALL DEPENDS ${CMAKE_BINARY_DIR}/shaders.metallib)
elseif(PIPELINE_LISTS)
  message(STATUS "No Metal compiler (xcrun metal): shaders will be compiled at run time")
endif()
