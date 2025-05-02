from setuptools import setup, find_packages
import versioneer, os, sys

SETUP_REQUIRES = ['setuptools>=30.3.0'] + (['wheel'] if 'bdist_wheel' in sys.argv else [])

with open(os.path.join(os.path.dirname(__file__), "requirements.txt")) as f:
    INSTALL_REQUIRES = f.read().splitlines()

setup(
    name='meg_qc',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    setup_requires=SETUP_REQUIRES,
    install_requires=INSTALL_REQUIRES,

    # ------------- AQUÍ EL CAMBIO ----------------------------
    packages=find_packages(include=['meg_qc', 'meg_qc.*']),
    # ----------------------------------------------------------

    include_package_data=True,                     # activa package_data
    package_data={'meg_qc.miscellaneous.GUI.other': ['logo.png']},

    entry_points={
        'console_scripts': [
            'megqc = meg_qc.miscellaneous.GUI.megqcGUI:run_megqc_gui',
            'run-megqc = meg_qc.test:run_megqc',
            'run-megqc-plotting = meg_qc.test:get_plots',
            'get-megqc-config = meg_qc.test:get_config',
        ]
    },
    url='https://github.com/karellopez/MEGqc',
    license='MIT',
    author='ANCP',
    author_email='karel.mauricio.lopez.vilaret@uni-oldenburg.de',
)
