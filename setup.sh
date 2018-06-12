#!/usr/bin/env bash

set -e

# TODO: Set to URL of git repo.
PROJECT_GIT_URL='http://git.yonyou.com/cuiymf/chatbot_test.git'

PROJECT_BASE_PATH='/home/workspace'


# Install Python
echo "Installing dependencies..."
wget https://www.python.org/ftp/python/3.6.4/Python-3.6.4.tar.xz
tar xvJf Python-3.6.4.tar.xz
cd Python-3.6.4
./configure
make
make install

# Install virtualenv
sudo apt-get install python-pip
sudo pip install virtualenv
mkdir ~/.virtualenvs
sudo pip install virtualenvwrapper
export WORKON_HOME=$HOME~/.virtualenvs
sudo source /usr/local/bin/virtualenvwrapper.sh
mkvirtualenv chat_bot

# Git clone the project and install dependencies
sudo mkdir -p $PROJECT_BASE_PATH
git clone $PROJECT_GIT_URL $PROJECT_BASE_PATH
pip install -r $PROJECT_BASE_PATH/requirements.txt
