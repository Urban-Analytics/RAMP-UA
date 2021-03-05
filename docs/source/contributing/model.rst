Contributing to the model code
===================================

We welcome contributions towards the model code. 
They will be considered by the core development team before being merged however we have a number of internal projects in active development
so please understand we may not include all community suggestions!


About model code
------------------------------

This model code is still in an active state of development so it is liable to change without significant notice.
Please be mindful of this if you choose to contribute and please ensure that any changes you suggest pass existing continuous integration steps.

Repository code checks
-------------------------

The RAMP-UA repository includes a number of GitHub actions that run automatically when changes are merged to the master branch or if pull requests
are suggested to master. 
These are included to ensure that changes do not break the existing model and all contributions will need to pass the test suite before they are accepted.

You can run the test suite locally using `pytest <https://docs.pytest.org/en/stable/>`__ which is included within the ``ramp-ua`` conda environment.
This should behave approximately the same within the GitHub actions test runners but occasionally differences can lead to errors so be sure to check 
the available GitHub actions logs carefully if you get unexpected test failures that can't be replicated locally. ::

    $ cd RAMP-UA-fork
    $ pytest tests/

This will run through the test suite and output details of information about whether tests have passed, failed and if warnings have been issued.
