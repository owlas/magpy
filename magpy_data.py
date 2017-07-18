from git import Repo
from git import Actor
import magpy
import os.path as osp
import pickle as pkl

def magpy_actor():
    return Actor("magpy-shelf", "github.com/owlas/magpy")

def shelve_results(results, repo_path, name):
    repo = Repo(repo_path)
    assert not repo.bare
    index = repo.index

    results_path = osp.join(repo.working_tree_dir, name)

    with open(results_path, 'wb') as f:
        if isinstance(results, magpy.EnsembleResults):
            pkl.dump(results.results, f)
        else:
            pkl.dump(results, f)

    index.add([results_path])
    index.commit("Added results ref:"+name, author=magpy_actor(), committer=magpy_actor())
    print('Shelved!')
    print('Shelf repo:', repo_path)
    print('Object name:', name)

def grab_results(repo_path, name):
    repo = Repo(repo_path)
    assert not repo.bare

    results_path = osp.join(repo.working_tree_dir, name)

    with open(results_path, 'rb') as f:
        res = pkl.load(f)

    if isinstance(res, list):
        results = magpy.EnsembleResults(res)
    else:
        results = res
    return results
