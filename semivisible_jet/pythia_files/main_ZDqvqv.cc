// ZDqvqv.cc
// Uses Version of Pythia8310
// Creates Scattering Events, Extracts Particle 4-Momentum, and Runs Anti-kt Jet Algorithm
// Output: 
//
// Input Specified in inputs.cmnd
//
// Requires Two Arguments:
// DMShower, DMHadron, DMDecay, SMShower, SMHadron
// Minimum Jet p_T
//
// Author: Kehang Bai, kbai@uoregon.edu

#include "Pythia8/Pythia.h"
#include "Pythia8Plugins/HepMC3.h"

using namespace Pythia8;



int main( int argc, char* argv[] ){
	
	// Check that correct number of command-line arguments
	if (argc != 3) {
	       	cerr << " Unexpected number of command-line arguments ("<<argc<<"). \n"
			<< " You are expected to provide the arguments" << endl
			<< " 1. Input file for settings" << endl
			<< " 2. Output file for HepMC events" << endl
			<< " Program stopped. " << endl;
		return 1;
	}

	// Specify file where HepMC events will be stored.
	Pythia8::Pythia8ToHepMC topHepMC(argv[2]);

	// Read in Commands and Extract Settings
	Pythia pythia;
	pythia.readString("HiddenValley:ffbar2Zv = on");
	
	// Input parameters:
        pythia.readFile(argv[1],0);
	int nEvent = pythia.mode("Main:numberOfEvents");

	// Run Pythia

	// Initialize Run
	pythia.readString("Random:setSeed = on");
	pythia.readString("Random:seed = 76");
	pythia.init();

        cout << endl << endl << endl;
        cout << "Start generating events" << endl;


	// Event Loop, Skips if Error Occurs and lists the first ten events
	for (int iEvent = 0; iEvent < nEvent; iEvent++) {
		
		if (!pythia.next()) continue;

		if (iEvent < 3) pythia.event.list();
		
		// Construct new empty HepMC event, fill it and write it out.
                topHepMC.writeNextEvent( pythia );
	}
		
	pythia.stat();
	return 0;
}

