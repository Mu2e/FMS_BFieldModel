# from __future__ import absolute_import
# import os
import git

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

# def get_conda_dir(cfgfile):
#     with open(cfgfile) as f:
#         directory = f.readlines()[0].strip()
#     return directory

# helicalc_dir = get_git_root('.')
# helicalc_dir = get_git_root(os.getcwd())
helicalc_dir = get_git_root(__file__)+'/'
# data directory (symbolic link)
helicalc_data = helicalc_dir + 'data/'
# location of Anaconda env directory
# conda_dir = get_conda_dir(helicalc_dir+'cfg/conda.cfg')
