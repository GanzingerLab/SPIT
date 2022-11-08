from setuptools import setup

setup(
    name = 'SPIT',
    description = 'A commandline app for single particle interaction tracking',
    license='MIT',
    version = '1.0.0',
    packages = ['spit'],
    author= 'Miles Henderson, Christian Niederauer',
    url='',
    author_email= 'c.niederauer@amolf.nl, m.wanghenderson@amolf.nl',
   classifiers=[
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
   ],
   install_requires=[
		"picasso @ git+https://github.com/jungmannlab/picasso.git#egg=picasso-0.3.1",
   ], 
)
