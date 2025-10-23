import os
import shutil
import sys
import subprocess
from dataclasses import dataclass
#from pydantic import BaseModel

@dataclass
class cfg_mujoco_c:
    src_dir:str
    build_dir:str
    install_dir:str
    build_type:str

@dataclass
class cfg_mujoco_py:
    build_type:str

@dataclass
class config:
    mujoco_c:cfg_mujoco_c
    mujoco_py:cfg_mujoco_py

def _run_cmd(cmd):
    full_cmd=' '.join(cmd)
    print(f"-- CMD : {full_cmd}")

    try:
        with subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text = True,
            bufsize = 1,
            encoding='utf-8',
            universal_newlines = True
        ) as proc:
            for line in proc.stdout:
                print(line, end='')

            proc.wait()

            if proc.returncode != 0:
                print(f"-- CMD END failed with exit code {proc.returncode}: {full_cmd} ", file=sys.stderr)
                sys.exit(proc.returncode)

    except FileNotFoundError:
        print(f"-- CMD END Command not found: {cmd[0]}: {full_cmd}", file=sys.stderr)
        sys.exit(127)  # 127 is standard POSIX code for "command not found"

    except PermissionError:
        print(f"-- CMD END Permission denied: {cmd[0]}: {full_cmd}", file=sys.stderr)
        sys.exit(126)  # 126 is standard POSIX code for "not executable"

    except OSError as e:
        print(f"-- CMD END OS error while trying to run command: {e}: {full_cmd}", file=sys.stderr)
        sys.exit(1)

    print(f"-- CMD END succeeded: {full_cmd}")


def __copytree(dst, src, symlinks = False, ignore = None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

def _clear_dir(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir, ignore_errors=True)

def _override_copy_dir(dst,src):
    _clear_dir(dst)
    if not os.path.exists(dst):
        os.mkdir(dst)
    __copytree(dst,src)

def _create_virtual_env_if_not_exists(dir):
    if not os.path.exists(dir):
        _run_cmd(['python', '-m', 'venv', dir])
        f = os.path.join(dir,'Scripts','activate')
        _run_cmd(['source', f])


def _cmake_install_mujoco_c(cfg_mujoco_c,current_dir):
    _clear_dir(cfg_mujoco_c.build_dir)

    src_flag='-S'+ cfg_mujoco_c.src_dir
    build_dir_flag='-B'+ cfg_mujoco_c.build_dir
    install_flag='-DCMAKE_INSTALL_PREFIX=' + cfg_mujoco_c.install_dir
    build_style3d_flag='-DMUJOCO_BUILD_STYLE3D=OFF'
    version_flag='-DCMAKE_POLICY_VERSION_MINIMUM=3.5'
    _run_cmd(['cmake', src_flag , build_dir_flag, install_flag,version_flag,build_style3d_flag])

    _clear_dir(cfg_mujoco_c.install_dir)
    _run_cmd(['cmake', '--build' ,cfg_mujoco_c.build_dir, '--target', 'install' , '--config',cfg_mujoco_c.build_type, ])

    mujoco_plugin_path = os.environ['MUJOCO_PLUGIN_PATH']
    _override_copy_dir(mujoco_plugin_path, os.path.join(cfg_mujoco_c.install_dir,'bin'))


def _pip_install_mujoco_py(cfg_mujoco_py,current_dir):
    _create_virtual_env_if_not_exists('.venv')

    _run_cmd(['sh', './make_sdist.sh'])
    _run_cmd(['pip', 'install', './dist/mujoco-3.3.6.tar.gz']) # make more rubost


def install(configs):
    current_dir = os.getcwd()
    for cfg in configs:
        _cmake_install_mujoco_c(cfg.mujoco_c, current_dir)
        _pip_install_mujoco_py(cfg.mujoco_py, current_dir)

configs = [ 
    config( cfg_mujoco_c('..','../temp_build','../temp_install','Release'), cfg_mujoco_py('Release'))
]

install(configs)