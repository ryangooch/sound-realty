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
- It would be possible to allow JSON files containing more than one sample at a time.
This was apparently outside of spec, however, so I chose to only allow one sample.
- Model caching could be done more efficiently; ideally perhaps, each model would be in its own
service, cached and available at need.
- Similarly, we should extract the data processing logic from the REST service if things get much
more complicated. Not that this implementation is complicated at all, but sequestering those
functions can make applications more easily maintainable and less complex.
- It is useful to include date information in the model; unfortunately, the test data does not
include date information. So instead we use an assumption that the goal is to find housing prices
today, though really what makes sense for this data is probably to use the last date seen in the
training data, since those dates are from 2014-15.
- Locally, I'm using `ruff` to provide type-checking and linting support. I also use Black on save
to format code. I prefer a wide coding canvas, so I have set the column width to 100 characters.