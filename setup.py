#!/usr/bin/env python
import sys, os
from setuptools import setup
import versioneer

# Give setuptools a hint to complain if it's too old a version
# 30.3.0 allows us to put most metadata in setup.cfg
# Should match pyproject.toml
SETUP_REQUIRES = ['setuptools >= 30.3.0']
# This enables setuptools to install wheel on-the-fly
SETUP_REQUIRES += ['wheel'] if 'bdist_wheel' in sys.argv else []
INSTALL_REQUIRES = []


if __name__ == '__main__':
    REQ_FILE_PATH = os.path.dirname(os.path.realpath(__file__)) + "/requirements.txt"
    with open(REQ_FILE_PATH) as f:
        INSTALL_REQUIRES = list(f.read().splitlines())
    setup(name='meg_qc',
        version= versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        setup_requires=SETUP_REQUIRES,
        install_requires=INSTALL_REQUIRES,
        packages=['meg_qc','meg_qc/calculation','meg_qc/calculation/metrics','meg_qc/plotting','meg_qc/settings',
                  'meg_qc/miscellaneous','meg_qc/miscellaneous/examples','meg_qc/miscellaneous/optimizations',
                  'meg_qc/miscellaneous/GUI','meg_qc/miscellaneous/GUI/other'
                  ],
    package_data={
        'meg_qc.miscellaneous.GUI.other': ['logo.png'],
    },
    include_package_data=True,

        url='https://github.com/karellopez/MEGqc',
        entry_points={
            'console_scripts':[
                'megqc = meg_qc.miscellaneous.GUI.megqcGUI:run_megqc_gui',
                'run-megqc = meg_qc.test:run_megqc',
                'run-megqc-plotting = meg_qc.test:get_plots',
                'get-megqc-config = meg_qc.test:get_config'
            ]  
        },
        license='MIT',
        author='ANCP',
        author_email='karel.mauricio.lopez.vilaret@uni-oldenburg.de')
