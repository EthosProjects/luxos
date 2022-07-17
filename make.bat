glslc Square.comp -o square.spv
call cl main.cpp /I include /std:c++20 -DDEBUGMODE=true /link lib/*.lib /OUT:debug.exe
echo Compiled successfully