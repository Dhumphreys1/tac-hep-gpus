#include "particle.h"

//------------------------------------------------------------------------------

//*****************************************************************************
//                                                                             *
//    MEMBERS functions of the Particle Class                                  *
//                                                                             *
//*****************************************************************************

//
//*** Default constructor ------------------------------------------------------
//
Particle::Particle(){
	pt = eta = phi = E = m = 0.0;
	p[0] = p[1] = p[2] = p[3] = 0.0;
}

//*** Additional constructor ------------------------------------------------------
Particle::Particle(double pt, double eta, double phi, double E){
	p[0] = this->pt = pt;
	p[1] = this->eta = eta;
	p[2] = this->phi = phi;
	p[3] = this->E = E;
}

//
//*** Members  ------------------------------------------------------
//
double Particle::sintheta(){
	// theta = 2 * arctan(exp(-eta))
    // sin(theta) = 2*exp(-eta) / (1 + exp(-2*eta))
    double eta = this->p[1]; // eta is stored in p[1]
    double result = 2.0 * std::exp(-eta) / (1.0 + std::exp(-2*eta));
    return result;
}

void Particle::p4(double pt, double eta, double phi, double E){

	// Not very descriptive method name... Set p4? Get p4?
	// Cant be get, its a void so I'm assuming set.
	p[0] = this->pt = pt;
	p[1] = this->eta = eta;
	p[2] = this->phi = phi;
	p[3] = this->E = E;

}

void Particle::setMass(double mass)
{
	this->m = mass;
}

//
//*** Prints 4-vector ----------------------------------------------------------
//
void Particle::print(){
	std::cout << std::endl;
	std::cout << "(" << p[0] <<",\t" << p[1] <<",\t"<< p[2] <<",\t"<< p[3] << ")" << "  " <<  sintheta() << std::endl;
}