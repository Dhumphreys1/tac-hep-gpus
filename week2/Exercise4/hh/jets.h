#include <particle.h>

class Jet : public Particle {
public:
    Jet() = default;
    Jet(double pt, double eta, double phi, double E) : Particle(pt, eta, phi, E) {}
    int flavor;
    int getFlavor() {return flavor;}
    void setFlavor(int newflavor){flavor = newflavor;}
    void print(){
        std::cout << std::endl;
        std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << "  " << m << "  " << flavor <<std::endl;
    };
};