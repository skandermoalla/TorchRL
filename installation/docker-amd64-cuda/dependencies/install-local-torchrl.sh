pushd "${PROJECT_ROOT_AT}"/repos/rl || exit
rm -rf dist/
rm -rf build/
rm -rf torchrl.egg-info
rm -rf torchrl_nightly.egg-info
python setup.py develop --user
popd || exit
