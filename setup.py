from setuptools import setup
import versioneer

setup(
    name='meg_qc',
    version= versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=['meg_qc'],
    url='https://github.com/ANCPLabOldenburg/MEGqc',
    license='MIT',
    author='ANCP',
    author_email='aaron.reer@uol.de',
)