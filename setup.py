from setuptools import setup, find_packages

USERNAME = 'magdalenafuentes'
NAME = 'tutorial'

with open('requirements.txt') as f:
    REQUIREMENTS = f.readlines()

setup(
    name='common',
    version='0.1.0',
    description='A Tutorial on Tempo, Beat and Downbeat estimation',
    long_description=open('README.md').read().strip(),
    long_description_content_type='text/markdown',
    author='Matthew E. P. Davies, Sebastian B\:ock, Magdalena Fuentes',
    author_email='mgfuenteslujambio@gmail.com',
    url='https://github.com/{}/{}'.format(USERNAME, NAME),
    packages=find_packages(),
    install_requires=open('requirements.txt').readlines(),
    extras_require={},
    keywords=['audio', 'beat', 'downbeat', 'music', 'tempo', 'ismir', 'tutorial'],
    license='CC BY-NC-SA 4.0',
)
