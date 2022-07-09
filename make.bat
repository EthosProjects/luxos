glslc Square.comp -o square.spv
call cl lib/*.lib main.cpp /I include /std:c++20
echo Compiled successfully