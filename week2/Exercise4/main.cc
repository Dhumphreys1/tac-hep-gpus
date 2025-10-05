#include <stdio.h>
#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "t1.h"
#include "lepton.h"
#include "particle.h"
#include "jets.h"

#include <TMath.h>
#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TCanvas.h>
#include <TLorentzVector.h>



// I'm not putting 3 classes and a dozen methods in a single script.
// See headers and source

int main() {

	/* ************* */
	/* Input Tree   */
	/* ************* */

	TFile *f      = new TFile("input.root","READ");
	TTree *t1 = (TTree*)(f->Get("t1"));

	// Read the variables from the ROOT tree branches
	t1->SetBranchAddress("lepPt",&lepPt);
	t1->SetBranchAddress("lepEta",&lepEta);
	t1->SetBranchAddress("lepPhi",&lepPhi);
	t1->SetBranchAddress("lepE",&lepE);
	t1->SetBranchAddress("lepQ",&lepQ);

	t1->SetBranchAddress("njets",&njets);
	t1->SetBranchAddress("jetPt",&jetPt);
	t1->SetBranchAddress("jetEta",&jetEta);
	t1->SetBranchAddress("jetPhi",&jetPhi);
	t1->SetBranchAddress("jetE", &jetE);
	t1->SetBranchAddress("jetHadronFlavour",&jetHadronFlavour);

	// Total number of events in ROOT tree
	Long64_t nentries = t1->GetEntries();

	for (Long64_t jentry=0; jentry<100;jentry++)
 	{
		t1->GetEntry(jentry);
		std::cout<<" Event "<< jentry <<std::endl;

		//FIX ME
		for (int i = 0; i < nleps; i++){
			Lepton lep(lepPt[i], lepEta[i], lepPhi[i], lepE[i]);
			lep.setCharge(lepQ[i]);
			lep.print();
		}

		for (int i = 0; i < njets; i++){
			Jet jet(jetPt[i], jetEta[i], jetPhi[i], jetE[i]);
			jet.setFlavor(jetHadronFlavour[i]);
			jet.print();
		}



	} // Loop over all events

  	return 0;
}
