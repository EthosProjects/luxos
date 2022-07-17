glslc Square.comp -o square.spv
call cl main.cpp /I include/ /std:c++20 /O2 /Fe:luxos.exe /link lib/*.lib /SUBSYSTEM:windows /ENTRY:mainCRTStartup
echo Compiled successfully