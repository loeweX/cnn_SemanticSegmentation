project(SemSeg)

cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)
find_package(Lua51 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/src)
include_directories(${CMAKE_CURRENT_BINARY_DIR})

include_directories(
  ${LUA_INCLUDE_DIR}
  ${Torch_INSTALL_INCLUDE}
)

add_executable(SemSeg
    src/main.cpp
)

set(lua_libs ${LUA_LIBRARIES} ${Boost_LIBRARIES} ${GOBJECT_LIBRARIES} -lm -ldl)

link_directories(${Torch_INSTALL_LIB})
target_link_libraries(SemSeg 
   ${OpenCV_LIBS} 
   ${lua_libs} 
   luaT 
   TH)
