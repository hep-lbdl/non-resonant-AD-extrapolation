# Makefile is a part of the PYTHIA event generator.
# Author: Kehang Bai, kbai@uoregon.edu
#
# This is is the Makefile used to build PYTHIA examples on POSIX systems.
# Example usage is:
#     make main01
# For help using the make command please consult the local system documentation,
# i.e. "man make" or "make --help".

################################################################################
# VARIABLES: Definition of the relevant variables from the configuration script.
################################################################################

# Set the shell.
SHELL=/usr/bin/env bash

# Include the configuration.
-include Makefile.inc

# Check distribution (use local version first, then installed version).
ifneq ("$(wildcard ../lib/libpythia8.*)","")
  PREFIX_LIB=../lib
  PREFIX_INCLUDE=../include
endif
CXX_COMMON:=-I$(PREFIX_INCLUDE) $(CXX_COMMON) $(GZIP_LIB)
CXX_COMMON+= -L$(PREFIX_LIB) -Wl,-rpath,$(PREFIX_LIB) -lpythia8 -ldl
PYTHIA=$(PREFIX_LIB)/libpythia8$(LIB_SUFFIX)

################################################################################
# RULES: Definition of the rules used to build the PYTHIA examples.
################################################################################

# Rules without physical targets (secondary expansion for specific rules).
.SECONDEXPANSION:
.PHONY: all clean

# All targets (no default behavior).
all:
	$(info Usage: make mainXX)

# PYTHIA library.
$(PYTHIA):
	$(error Error: PYTHIA must be built, please run "make"\
                in the top PYTHIA directory)

# HEPMC3
main_ZDqvqv:\
	$(PYTHIA) $$@.cc
ifeq ($(HEPMC3_USE),true)
	$(CXX) $@.cc -o $@ $(CXX_COMMON) $(HEPMC3_INCLUDE) $(HEPMC3_LIB)\
	 -DHEPMC3
else
	$(error Error: $@ requires HEPMC3)
endif


# Plugin libraries.
lib%.so: %.cc $(PYTHIA)
	$(CXX) $< -o $@ -w $(CXX_COMMON) $(CXX_SHARED) -Wl,--no-as-needed


