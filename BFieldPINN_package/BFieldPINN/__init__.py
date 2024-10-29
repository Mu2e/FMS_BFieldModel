import os
import git

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

BFieldPINN_dir = os.path.join(get_git_root(__file__), '')
# data dir is symbolic link
BFieldPINN_data = os.path.join(BFieldPINN_dir, 'data', '')
