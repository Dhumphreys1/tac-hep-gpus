#include "particle.h"

class Lepton : public Particle {
public:
    Lepton() = default;
    Lepton(double pt, double eta, double phi, double E) : Particle(pt, eta, phi, E) {}
    int charge = 0;
    int getCharge() {return charge;}
    void setCharge(int q) { charge = q; }
};