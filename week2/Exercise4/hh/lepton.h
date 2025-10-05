#include "particle.h"

class Lepton : public Particle {
public:
    Lepton() = default;
    Lepton(double pt, double eta, double phi, double E) : Particle(pt, eta, phi, E) {}
    int charge = 0;
    int getCharge() {return charge;}
    void setCharge(int q) { charge = q; }
    void print(){
        std::cout << std::endl;
        std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << "  " << m << "  " << charge <<std::endl;
    };
};