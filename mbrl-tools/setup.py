import setuptools

setuptools.setup(
    name="mbrl-tools",
    version="0.1.dev0",
    description=(
        "RL tools to evaluate model based reinforcement learning solutions."
    ),
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/ramp-kits/rl_simulator/mbrl-tools",
    packages=setuptools.find_packages(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    entry_points={
        'console_scripts': [
            'mbrl-run = mbrltools.model_based_rl:mbrl_run_command',
            'mbrl-simulate = mbrltools.simulate_traces:simulate_command',
        ],
    },
    python_requires='>=3.6',
)
