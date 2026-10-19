Artifact-write failure, not an inference failure. At warm-repeat-22 Python's open(results.json, 'w') returned
EINVAL. The engine then exited normally through QUIT; the final record retry from close() saved its results.
The completed requests show stable 281 execution owners and a working 255-slot adaptive cache, but the disk
session/restart portion did not run. This is not an end-to-end acceptance pass.

The benchmark now retries only EINVAL/EACCES artifact writes, five bounded attempts outside timed generation.
The complete rerun is in ../verified/; that result supersedes this partial run.
