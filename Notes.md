# Implementation and Project Notes
- Initialized the `git` repo through the web interface then cloned locally
- Used `docker init` to initialize the Docker files
- Initially copied an old Dockerfile I had but the `conda` environment differed from the other
    file's virtual environment
    - Used a combination of Copilot and outside references to get `conda` environment running
    - Refer to [Activate Conda Dockerfile](https://pythonspeed.com/articles/activate-conda-dockerfile/)
- This *could* be its own repo with only the environment info needed to serve the model and its
predictions via the REST interface.
However, we also need to do a little data science/model tuning, so I'm going to add some standard
data science stack libraries and include a directory for a notebook, for visualization and workflow
preservation.