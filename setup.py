#!/usr/bin/env python

from distutils.core import setup

setup(name='non-resonant-AD',
      version='1.0',
      packages=['helpers', 'scripts', 'semivisible_jet'],
      entry_points={
          'console_scripts': [
              'run-reweighting = scripts.run_reweighting:main',
              'run-CATHODE = scripts.run_CATHODE:main',
              'run-FETA = scripts.run_FETA:main',
              'run-SALAD = scripts.run_SALAD:main',
              'run-idealAD = scripts.run_idealAD:main',
              'run-supervised = scripts.run_supervised:main',
              'run-trainAD = scripts.run_trainAD:main',
              'run-evaAD = scripts.run_evaluateAD:main',
              'plt-sig-inj = scripts.plot_sig_inj:main',
              'plt-avg-SIC = scripts.plot_avg_max_SIC:main',
              'plt-multi-SIC = scripts.plot_multi_max_SIC:main',
              'gen-toy-ds = scripts.gen_siginj_toy_dataset:main',
              'gen-phys-ds = scripts.gen_siginj_phys_dataset:main',
              'gen-phys-id-ds = scripts.gen_siginj_phys_ideal_dataset:main',
              'gen-phys-ts = scripts.gen_siginj_phys_testset:main',
          ],
      },
      )