#include <iostream>
#include <iterator>

void swapArrays(int *a, int *b, int arrayLength){
    int temp;
    for (int i = 0; i < arrayLength; i++){
        temp = a[i];
        a[i] = b[i];
        b[i] = temp;
    }
}

int main () {
    int a[10];
    int b[10];

    // Fill dummy arrays, a: 0-9, b: 10-19
    for (int i = 0; i < 10; i++){
        a[i] = i;
        b[i] = i + 10;
    }

    if (std::size(a) != std::size(b)){
        std::cout << "Arrays do not have the same dimension. Cannot swap." << std::endl;
        return 0;
    }

    swapArrays(a, b, std::size(a));
    std::cout << "Printing A elements" << std::endl;
    for (int element : a){
        std::cout<< element <<std::endl;
    }
    std::cout << "Printing B elements" << std::endl;
    for (int element : b){
        std::cout<< element <<std::endl;
    }

    return 0;
}