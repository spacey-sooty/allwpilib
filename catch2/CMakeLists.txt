include(FetchContent)

fetchcontent_declare(
    catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG
        4e8d92bf02f7d1c8006a0e7a5ecabd8e62d98502 # 3.6.0
)

fetchcontent_getproperties(catch2)
if(NOT catch2_POPULATED)
    fetchcontent_populate(catch2)

    add_subdirectory(${catch2_SOURCE_DIR} ${catch2_BINARY_DIR} EXCLUDE_FROM_ALL)
endif()
