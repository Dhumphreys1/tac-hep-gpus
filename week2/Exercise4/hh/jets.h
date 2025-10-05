#include <particle.h>

class Jet : public Particle {
public:
    Jet() = default;
    Jet(double pt, double eta, double phi, double E) : Particle(pt, eta, phi, E) {}
    int flavor;
    int getFlavor() {return flavor;}
    void setFlavor(int newflavor){flavor = newflavor;}
};