//------------------------------------------------------------------------------
// Particle Class
//

#ifndef PARTICLE_H
#define PARTICLE_H
#include <iostream>
#include <math.h>

class Particle{
	public:
	Particle();
	Particle(double pt, double eta, double phi, double E);
	double   pt, eta, phi, E, m, p[4];
	void     p4(double pt, double eta, double phi, double E);
	void     print();
	void     setMass(double);
	double   sintheta();
};

#endif