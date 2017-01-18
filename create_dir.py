import os

def create_dir(dirname):
    if not os.path.exists(dirname):
        try:
            os.makedirs(dirname)
        except OSError, e:
            if e.errno != 17: # dir already exists (occurs due to MPI)
                raise
            else:
                pass
