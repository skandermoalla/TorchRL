pushd repos/rl || exit
rm -rf dist/
rm -rf build/
rm -rf torchrl.egg-info
rm -rf torchrl_nightly.egg-info
python setup.py develop
popd || exit
