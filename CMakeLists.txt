cmake_minimum_required(VERSION 3.15)
project(window_capture LANGUAGES CXX)

# Enable C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find numpy
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy; print(numpy.get_include())"
    OUTPUT_VARIABLE NUMPY_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find OpenCV
find_package(OpenCV REQUIRED core)

# obs-game-capture-lib sources
set(OBS_GAME_CAPTURE_LIB_SRC
    obs-game-capture-lib/game_capture.cpp
)

# Main extension source
set(WINDOW_CAPTURE_SRC
    src/window_capture.cpp
)

# Extension module name
set(MODULE_NAME window_capture)

# Build the Python extension
add_library(${MODULE_NAME} MODULE ${WINDOW_CAPTURE_SRC} ${OBS_GAME_CAPTURE_LIB_SRC})

# Include directories
include_directories(
    ${Python3_INCLUDE_DIRS}
    ${NUMPY_INCLUDE_DIR}
    obs-game-capture-lib
    obs-game-capture-lib/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link libraries
if(WIN32)
    target_link_libraries(${MODULE_NAME} PRIVATE user32 gdi32)
endif()
target_link_libraries(${MODULE_NAME} PRIVATE ${Python3_LIBRARIES} ${OpenCV_LIBS})

# Set output name to .pyd for Python extension on Windows
if(WIN32)
    set_target_properties(${MODULE_NAME} PROPERTIES SUFFIX ".pyd")
endif()

