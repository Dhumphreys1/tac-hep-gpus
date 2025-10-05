# How to run main.cc

First we have to compile :
```
 g++ -std=c++17 -I hh -o main main.cc src/*.cxx `root-config --cflags --glibs`
```

And then excecute:
```
./main
```