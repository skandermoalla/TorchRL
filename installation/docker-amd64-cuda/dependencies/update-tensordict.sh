pushd repos/tensordict || exit
rm -rf dist/
rm -rf build/
rm -rf tensordict.egg-info
rm -rf tensordict_nightly.egg
python setup.py develop
popd || exit
