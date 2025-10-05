#include <string>
#include <vector>
#include <iostream>
struct TacHeper {
    std::string email;
    std::string name;
    std::string experiment;
};

void printStudentDetails(const TacHeper &student){
        std::cout <<"Name: " << student.name << std::endl;
        std::cout <<"Email: " <<student.email << std::endl;
        std::cout <<"Experiment: "<< student.experiment <<std::endl;
}

int main(){
    
    std::vector<std::string> emails{"dhumphreys@umass.edu"};
    std::vector<std::string> names{"Daniel Humphreys"};
    std::vector<std::string> experiments{"ATLAS"};
    std::vector<TacHeper> TacHepers{};

    for (int i = 0; i < emails.size(); i++){
        TacHeper student;
        student.email = emails[i];
        student.name = names[i];
        student.experiment = experiments[i];
        TacHepers.push_back(student);
    }

    for (const TacHeper &student : TacHepers){
        printStudentDetails(student);
    }

    return 0;
}