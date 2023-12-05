# PYTHIA configuration file.
# Generated on Wed 27 Sep 2023 02:13:57 PM PDT with the user supplied options:
# --with-hepmc3=/global/homes/k/kbai/Software/hepmc3-install
# --with-hepmc3-lib=/global/homes/k/kbai/Software/hepmc3-install/lib64
# --with-fastjet3=/global/homes/k/kbai/Software/fastjet-install

# Install directory prefixes.
PREFIX_BIN=/global/homes/k/kbai/Software/pythia8310/bin
PREFIX_INCLUDE=/global/homes/k/kbai/Software/pythia8310/include
PREFIX_LIB=/global/homes/k/kbai/Software/pythia8310/lib
PREFIX_SHARE=/global/homes/k/kbai/Software/pythia8310/share/Pythia8

# Compilation flags (see ./configure --help for further documentation).
CXX=g++
CXX_COMMON=-O2 -std=c++11 -pedantic -W -Wall -Wshadow -fPIC -pthread
CXX_SHARED=-shared
CXX_SONAME=-Wl,-soname,
LIB_SUFFIX=.so
OBJ_COMMON=

EVTGEN_USE=false
EVTGEN_CONFIG=
EVTGEN_BIN=
EVTGEN_INCLUDE=
EVTGEN_LIB=

FASTJET3_USE=true
FASTJET3_CONFIG=fastjet-config
FASTJET3_BIN=/global/u1/k/kbai/Software/fastjet-install/bin/
FASTJET3_INCLUDE=-I/global/u1/k/kbai/Software/fastjet-install/include
FASTJET3_LIB=-L/global/u1/k/kbai/Software/fastjet-install/lib -Wl,-rpath,/global/u1/k/kbai/Software/fastjet-install/lib -lfastjet

HEPMC2_USE=false
HEPMC2_CONFIG=
HEPMC2_BIN=
HEPMC2_INCLUDE=
HEPMC2_LIB=

HEPMC3_USE=true
HEPMC3_CONFIG=HepMC3-config
HEPMC3_BIN=/global/u1/k/kbai/Software/hepmc3-install/bin/
HEPMC3_INCLUDE=-I/global/u1/k/kbai/Software/hepmc3-install/include
HEPMC3_LIB=-L/global/u1/k/kbai/Software/hepmc3-install/lib64 -Wl,-rpath,/global/u1/k/kbai/Software/hepmc3-install/lib64 -lHepMC3

LHAPDF5_USE=false
LHAPDF5_CONFIG=
LHAPDF5_BIN=
LHAPDF5_INCLUDE=
LHAPDF5_LIB=

LHAPDF6_USE=false
LHAPDF6_CONFIG=
LHAPDF6_BIN=
LHAPDF6_INCLUDE=
LHAPDF6_LIB=

POWHEG_USE=false
POWHEG_CONFIG=
POWHEG_BIN=
POWHEG_INCLUDE=
POWHEG_LIB=

RIVET_USE=false
RIVET_CONFIG=
RIVET_BIN=
RIVET_INCLUDE=
RIVET_LIB=

ROOT_USE=false
ROOT_CONFIG=
ROOT_BIN=
ROOT_INCLUDE=
ROOT_LIB=

GZIP_USE=false
GZIP_CONFIG=
GZIP_BIN=
GZIP_INCLUDE=
GZIP_LIB=

PYTHON_USE=false
PYTHON_CONFIG=
PYTHON_BIN=
PYTHON_INCLUDE=
PYTHON_LIB=

MG5MES_USE=false
MG5MES_CONFIG=
MG5MES_BIN=
MG5MES_INCLUDE=
MG5MES_LIB=

OPENMP_USE=false
OPENMP_CONFIG=
OPENMP_BIN=
OPENMP_INCLUDE=
OPENMP_LIB=

MPICH_USE=false
MPICH_CONFIG=
MPICH_BIN=
MPICH_INCLUDE=
MPICH_LIB=

HDF5_USE=false
HDF5_CONFIG=
HDF5_BIN=
HDF5_INCLUDE=
HDF5_LIB=

HIGHFIVE_USE=false
HIGHFIVE_CONFIG=
HIGHFIVE_BIN=
HIGHFIVE_INCLUDE=
HIGHFIVE_LIB=
