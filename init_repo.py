from subprocess import call
import os

os.chdir("FlYMazeRL")

# Make sure the repository is up to date
call(["git", "pull"])

