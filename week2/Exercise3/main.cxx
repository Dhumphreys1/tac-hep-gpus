#include <iostream>

void playRockPaperScissors(int a, int b){
    if (a > 3 || a < 1){
        std::cout<< "Player A used an invalid move! Try again!"<<std::endl;
        return;
    }
    if (b > 3 || b < 1){
        std::cout<< "Player A used an invalid move! Try again!"<<std::endl;
        return;
    }
    std::string states[3] = {"rock", "paper", "scissors"};
    std::cout<< "Player A plays: " << states[a-1] <<std::endl;
    std::cout<< "Player B plays: " << states[b-1] <<std::endl;
    int result;

    // typical result
    if (a < b){
        result = 2;
    }
    else if (a > b){
        result = 1;
    }
    else{
        result = 0;
    }

    // handle the cyclic bit
    if (a==1 && b==3){
        result = 1;
    }
    if (a==3 && b==1){
        result = 2;
    }

    switch (result){
        case 0:
            std::cout << "Its a tie!" <<std::endl;
            break;
        case 1:
            std::cout<< "Player A wins!" <<std::endl;
            break;
        case 2:
            std::cout << "Player B wins!" <<std::endl;
            break;
    }
}

int main (){
    // I choose to use numbers to model the states.
    // To use strings I would just make a map from
    // strings to numbers and keep the same function.

    // rock = 1
    // paper = 2
    // scissors = 3
    int a = 1;
    int b = 1;
    playRockPaperScissors(a, b);
}