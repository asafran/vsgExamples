set(SOURCES
        AtmoshpereConstatnts.h
        AtmosphereTools.h
        Atmosphere.h
        Atmosphere.cpp
        AtmosphereLighting.h
        AtmosphereLighting.cpp
        InverseMatrices.h
        vsgatmosphere.cpp)

add_executable(vsgatmosphere ${SOURCES})

target_link_libraries(vsgatmosphere vsg::vsg)

if (vsgXchange_FOUND)
    target_compile_definitions(vsgatmosphere PRIVATE vsgXchange_FOUND)
    target_link_libraries(vsgatmosphere vsgXchange::vsgXchange)
endif()

install(TARGETS vsgatmosphere RUNTIME DESTINATION bin)
