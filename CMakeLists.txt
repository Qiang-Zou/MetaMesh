cmake_minimum_required(VERSION 3.18)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

project(3DPrintPlanner LANGUAGES CXX CUDA)

find_package(Qt6 REQUIRED COMPONENTS Gui Core Xml)
find_package(Boost REQUIRED
             COMPONENTS system filesystem thread)
find_package(OpenGL REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TBB REQUIRED)

include_directories(.)
include_directories(./QMeshLib)
include_directories(./QMeshLib/OpenMesh)
include_directories(./GLKLib)
include_directories(./GLUILib/include)
include_directories(./Utils)
include_directories(./LatticeTri/include)

link_directories(/usr/local/lib)
link_directories(/usr/lib/x86_64-linux-gnu)

#########################################################################
file(GLOB MAIN_SOURCE_FILES "${CMAKE_SOURCE_DIR}/*.cpp")
file(GLOB MAIN_HEADER_FILES "${CMAKE_SOURCE_DIR}/*.h")
file(GLOB_RECURSE QMESHLIB_SOURCE "${CMAKE_SOURCE_DIR}/QMeshLib/*.cpp" "${CMAKE_SOURCE_DIR}/QMeshLib/*.cc")
file(GLOB_RECURSE QMESHLIB_HEADER "${CMAKE_SOURCE_DIR}/QMeshLib/*.h" "${CMAKE_SOURCE_DIR}/QMeshLib/*.hh")
file(GLOB_RECURSE UTILS_SOURCE "${CMAKE_SOURCE_DIR}/Utils/*/*.cpp")
file(GLOB_RECURSE UTILS_HEADER "${CMAKE_SOURCE_DIR}/Utils/*/*.h")
file(GLOB_RECURSE GLK_SOURCE "${CMAKE_SOURCE_DIR}/GLKLib/*.cpp")
file(GLOB_RECURSE GLK_HEADER "${CMAKE_SOURCE_DIR}/GLKLib/*.h")
file(GLOB_RECURSE PQP_SOURCE "${CMAKE_SOURCE_DIR}/PQPLib/*.cpp")
file(GLOB_RECURSE PQP_HEADER "${CMAKE_SOURCE_DIR}/PQPLib/*.h")
file(GLOB_RECURSE GLUI_SOURCE_1 "${CMAKE_SOURCE_DIR}/GLUILib/*.c")
file(GLOB_RECURSE GLUI_SOURCE_2 "${CMAKE_SOURCE_DIR}/GLUILib/*.cpp")
file(GLOB_RECURSE GLUI_HEADER "${CMAKE_SOURCE_DIR}/GLUILib/*.h")
file(GLOB TRI_HEADER 
    "${CMAKE_SOURCE_DIR}/LatticeTri/include/*.cuh"
    "${CMAKE_SOURCE_DIR}/LatticeTri/include/*.h"
)
file(GLOB TRI_SOURCE 
    "${CMAKE_SOURCE_DIR}/LatticeTri/src/*.cu"
    "${CMAKE_SOURCE_DIR}/LatticeTri/src/*.cpp"
)

add_executable(${PROJECT_NAME} 
    ${MAIN_SOURCE_FILES} 
    ${MAIN_HEADER_FILES} 
    ${QMESHLIB_SOURCE} 
    ${QMESHLIB_HEADER} 
    ${UTILS_SOURCE} 
    ${UTILS_HEADER} 
    ${GLK_SOURCE} 
    ${GLK_HEADER} 
    ${PQP_SOURCE} 
    ${PQP_HEADER}
    ${GLUI_SOURCE_1} 
    ${GLUI_SOURCE_2} 
    ${GLUI_HEADER}
    ${TRI_HEADER}
    ${TRI_SOURCE}
)

target_link_libraries(${PROJECT_NAME} GLU glut GLEW)
target_link_libraries(${PROJECT_NAME} Qt6::Core Qt6::Gui Qt6::Xml)
target_link_libraries(${PROJECT_NAME} OpenGL::GL)
target_link_libraries(${PROJECT_NAME} gmp mpfr boost_thread)
target_link_libraries(${PROJECT_NAME} tbb)

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)
