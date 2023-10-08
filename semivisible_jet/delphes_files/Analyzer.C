/*
Simple macro showing how to access branches from the delphes output root file,
loop over events, and plot simple quantities such as the jet pt.

root -l examples/Analyzer.C'("delphes_output.root", "output.txt")'
*/

#ifdef __CLING__
R__LOAD_LIBRARY(libDelphes)
#include "classes/DelphesClasses.h"
#include "external/ExRootAnalysis/ExRootTreeReader.h"
#else
class ExRootTreeReader;
class ExRootResult;
#endif

//------------------------------------------------------------------------------


void AnalyseEvents(ExRootTreeReader *treeReader,  const char *outputFile_part)
{

  // Get pointers to branches used in this analysis
  TClonesArray *branchEvent = treeReader->UseBranch("Event");
  TClonesArray *branchJet = treeReader->UseBranch("FatJet");
  TClonesArray *branchScalarHT = treeReader->UseBranch("ScalarHT");
  TClonesArray *branchMET = treeReader->UseBranch("MissingET");

  Long64_t allEntries = treeReader->GetEntries();
  
  ofstream myfile_part;

  cout << "** Chain contains " << allEntries << " events" << endl;

  myfile_part.open (outputFile_part);
  
  myfile_part << "pT_j1"<< " " << "eta_j1" << " " << "phi_j1" << " " << "pT_j2" << " " << "eta_j2" << " " << "phi_j2" << " " << "m_jj" << " " << "tau21_j1" << " " << "tau21_j2" << " " << "tau32_j1" << " " << "tau32_j2" << " " << "met" << " " << "phi_met" << " " << "min_dPhi" << " " << "ht" << " " << std::endl;

  int nEvent0jets = 0;
  int nEvent1jets = 1;
  int nEvent2jets = 2;

  // Loop over all events
  for(Int_t entry = 0; entry < allEntries; ++entry)
  {
    // Load selected branches with data from specified event
    treeReader->ReadEntry(entry);
    
    if(entry%1000 == 0) cout << "Event number: "<< entry <<endl;


    Float_t pT_j1, pT_j2, eta_j1, eta_j2, phi_j1, phi_j2; 
    Float_t tau21_j1, tau21_j2, tau32_j1, tau32_j2; 
    Float_t m_jj, met, phi_met, ht;

    Jet *jet1, *jet2;
    MissingET *MET1;
    ScalarHT *HT1; 

    if(branchJet->GetEntries() == 0) nEvent0jets++;
    if(branchJet->GetEntries() == 1) nEvent1jets++;
    if(branchJet->GetEntries() == 2) nEvent2jets++;
    

    // If event contains at least 2 jet
    if(branchJet->GetEntries() > 1)
    {
      // Take first 2 jets
      jet1 = (Jet*) branchJet->At(0);
      jet2 = (Jet*) branchJet->At(1);
      
      // Take HT
      HT1 = (ScalarHT*) branchScalarHT->At(0);
      
      // Take Met
      MET1 = (MissingET *) branchMET->At(0);
      
      // Save quantities

      pT_j1 = jet1->PT;
      eta_j1 = jet1->Eta;
      phi_j1 = jet1->Phi;

      pT_j2 = jet2->PT;
      eta_j2 = jet2->Eta;
      phi_j2 = jet2->Phi;

      tau21_j1 = jet1->Tau[1]/jet1->Tau[0];
      tau21_j2 = jet2->Tau[1]/jet2->Tau[0];

      tau32_j1 = jet1->Tau[2]/jet1->Tau[1];
      tau32_j2 = jet2->Tau[2]/jet2->Tau[1];

      met = MET1->MET;
      phi_met = MET1->Phi;

      ht = HT1->HT;

      // Calculate invariant mass
      m_jj = ((jet1->P4()) + (jet2->P4())).M();
      
      // Calculate min delta Phi
      Float_t dPhi_j1 = fabs(phi_met - phi_j1);
      Float_t dPhi_j2 = fabs(phi_met - phi_j2);
      
      // Ensure the angle is within the range [0, pi)
      while (dPhi_j1 >= M_PI)
          dPhi_j1 -= M_PI;
      while (dPhi_j2 >= M_PI)
          dPhi_j2 -= M_PI;
      
      Float_t min_dPhi = std::min(dPhi_j1, dPhi_j2);



      myfile_part << pT_j1 << " " << eta_j1 << " " << phi_j1 << " " << pT_j2 << " " << eta_j2 << " " << phi_j2 << " " << m_jj << " " << tau21_j1 << " " << tau21_j2 << " " << tau32_j1 << " " << tau32_j2 << " " << met << " " << phi_met << " " << min_dPhi << " " << ht << " "<< std::endl;

    } 
  }

  cout << "nEvent0jets: " << nEvent0jets << std::endl;
  cout << "nEvent1jets: " << nEvent1jets << std::endl;
  cout << "nEvent2jets: " << nEvent2jets << std::endl;

}

//------------------------------------------------------------------------------

void Analyzer(const char *inputFile, const char *outputFile_part)
{
  gSystem->Load("libDelphes");

  TChain *chain = new TChain("Delphes");
  chain->Add(inputFile);

  ExRootTreeReader *treeReader = new ExRootTreeReader(chain);

  AnalyseEvents(treeReader, outputFile_part);

  cout << "** Exiting..." << endl;

  delete treeReader;
  delete chain;
}

//------------------------------------------------------------------------------

