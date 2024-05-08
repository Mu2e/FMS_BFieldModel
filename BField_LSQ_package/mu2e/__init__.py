from __future__ import absolute_import
import os
import git
# mu2e_ext_path = os.path.expanduser('~/Coding/Mu2E_Extras/')
#mu2e_ext_path = os.path.expanduser('~/data/')

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

mu2e_dir = os.path.join(get_git_root(__file__), 'BField_LSQ_package', '')
mu2e_ext_path = os.path.join(mu2e_dir, 'data', '')
